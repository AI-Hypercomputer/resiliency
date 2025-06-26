"""Copyright 2025 Google LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from collections import ChainMap
import os
import dataclasses
import io
from pathlib import Path
from time import time
from typing import Any, Dict, List, Optional, Set, Tuple, Union, cast

from lightning.fabric.utilities.cloud_io import get_filesystem
from lightning.fabric.utilities.types import _PATH
from megatron.core.dist_checkpointing.core import CheckpointingException
from megatron.core.dist_checkpointing.dict_utils import (
    extract_matching_values,
    merge,
    nested_values,
)
from megatron.core.dist_checkpointing.mapping import (
    ShardedObject,
    ShardedStateDict,
    ShardedTensor,
    StateDict,
    apply_factory_merges,
)
from megatron.core.dist_checkpointing.serialization import (
    get_default_load_sharded_strategy,
    get_default_save_sharded_strategy,
)
from megatron.core.dist_checkpointing.state_dict_transformation import load_preprocess
from megatron.core.dist_checkpointing.strategies import tensorstore
from megatron.core.dist_checkpointing.strategies.async_utils import AsyncRequest
from megatron.core.dist_checkpointing.strategies.base import (
    LoadCommonStrategy,
    LoadShardedStrategy,
)
from megatron.core.dist_checkpointing.strategies.common import COMMON_STATE_FNAME
from megatron.core.dist_checkpointing.strategies.filesystem_async import (
    FileSystemWriterAsync,
)
from megatron.core.dist_checkpointing.strategies.fully_parallel import (
    FullyParallelLoadStrategyWrapper,
    FullyParallelSaveStrategyWrapper,
)
from megatron.core.dist_checkpointing.strategies.resharding import (
    TensorReformulationMetadata,
    apply_nd_flattened_tensors_reformulation,
    is_nd_flattened_tensor,
    restore_nd_flattened_tensors_formulation,
)
from megatron.core.dist_checkpointing.strategies.torch import (
    MCoreLoadPlanner,
    MCoreSavePlanner,
    TorchDistLoadShardedStrategy,
    TorchDistSaveShardedStrategy,
    _replace_sharded_keys_with_state_dict_keys,
    _replace_state_dict_keys_with_sharded_keys,
    _restore_dict_types,
    _unwrap_pyt_sharded_tensor,
    mcore_to_pyt_state_dict,
)
from megatron.core.dist_checkpointing.utils import extract_sharded_base
from megatron.core.dist_checkpointing.validation import (
    StrictHandling,
    determine_global_metadata,
    parse_strict_flag,
    validate_integrity_and_strict_load,
    validate_sharded_objects_handling,
    verify_checkpoint_and_load_strategy,
)
from megatron.core.parallel_state import get_data_parallel_group
from nemo.utils import logging
from nemo.utils.callbacks.dist_ckpt_io import DistributedCheckpointIO
from resiliency.third_party.megatron.async_utils import AsyncRequest, debug_time
from resiliency.utils import get_resiliency_logger
import torch
from torch import distributed as dist
from torch.distributed._shard.sharded_tensor import ShardedTensor as TorchShardedTensor
from torch.distributed.checkpoint import (
    DefaultLoadPlanner,
    LoadPlan,
    Metadata,
    SavePlan,
)
from torch.distributed.checkpoint._dedup_save_plans import dedup_save_plans
from torch.distributed.checkpoint.default_planner import create_default_global_save_plan
from torch.distributed.checkpoint.filesystem import _StoragePrefix
from torch.distributed.checkpoint.logger import _dcp_method_logger
from torch.distributed.checkpoint.metadata import STATE_DICT_TYPE
from torch.distributed.checkpoint.planner import LoadPlanner, SavePlanner
from torch.distributed.checkpoint.storage import StorageReader
from torch.distributed.checkpoint.utils import _DistWrapper

from ._ckpt_utils import get_is_checkpoint_file_handler, save
from .replication_utils import (
    ReplicationCoordinator,
    ReplicationLoadPlanner,
    ReplicationStorageReader,
    get_local_ckpt_rank,
    get_replication_coordinator,
)

logger = get_resiliency_logger(__name__)


class InClusterLocalCheckpointIO(DistributedCheckpointIO):

  def __init__(
      self,
      save_ckpt_format: str,
      load_directly_on_device: bool = True,
      load_strictness: Optional["StrictHandling"] = None,
      async_save: bool = False,
      torch_dist_multiproc: Optional[int] = None,
      assume_constant_structure: bool = False,
      parallel_save: bool = False,
      parallel_save_within_dp: bool = False,
      parallel_load: bool = False,
      use_ckpt_load_replication: bool = False,
  ):
    super().__init__(
        save_ckpt_format,
        load_directly_on_device,
        load_strictness,
        async_save,
        torch_dist_multiproc,
        assume_constant_structure,
        parallel_save,
        parallel_save_within_dp,
        parallel_load,
    )

    self.use_ckpt_load_replication = use_ckpt_load_replication
    self.replication_coordinator = None

  @debug_time("InClusterLocalCheckpointIO.save_checkpoint")
  def save_checkpoint(
      self,
      checkpoint: Dict[str, Any],
      path: _PATH,
      storage_options: Optional[Any] = None,
  ) -> Optional["AsyncRequest"]:
    """Saves a distributed checkpoint.

    Creates the checkpoint root directory if doesn't exist.

    Args:
        checkpoint (Dict[str, Any]): sharded state dict to save
        path (_PATH): checkpoint directory
        storage_options (Any, optional): Optional parameters when saving the
          checkpoint
    """
    use_in_cluster_local_ckpts=True
    is_persistent_storage=storage_options.get("is_persistent_storage", False)

    fs = get_filesystem(path)
    if get_is_checkpoint_file_handler(
      use_in_cluster_local_ckpts,
      is_persistent_storage
    ):
      fs.makedirs(path, exist_ok=True)
    dist.barrier()

    validate_sharding_integrity = not (
        self.validated_consistency and self.assume_constant_structure
    )
    self.validated_consistency = True
    return save(
        sharded_state_dict=checkpoint,
        checkpoint_dir=path,
        sharded_strategy=self.save_sharded_strategy,
        validate_access_integrity=validate_sharding_integrity,
        async_sharded_save=self.async_save,
        use_in_cluster_local_ckpts=use_in_cluster_local_ckpts,
        is_persistent_storage=is_persistent_storage,
    )

  @debug_time("InClusterLocalCheckpointIO.load_checkpoint")
  def load_checkpoint(
      self,
      path: _PATH,
      map_location: Optional[Any] = None,
      sharded_state_dict: Dict[str, Any] = None,
      strict: Union[None, bool, "StrictHandling"] = None,
      validate_access_integrity: Optional[bool] = True,
  ) -> Dict[str, Any]:
    """Loads a distributed checkpoint.

    Args:
        path (_PATH): checkpoint directory
        map_location (Any, optional): required to be None in this implementation
        sharded_state_dict (Dict[str, Any], optional): state dict which defines
          the loading procedure for the distributed checkpoint. Defaults to None
          to comply with the CheckpointIO interface, but it's a required
          argument.
        strict (bool, StrictHandling, optional): adjust load strictness. bool
          value is translated to StrictHandling instance. Gets overwritten by
          `self.load_strictness`. Defaults to None. If `self.load_strictness` is
          also None, strict becomes StrictHandling.ASSUME_OK_UNEXPECTED.

        Returns:
            Dist[str, Any]: loaded checkpoint.
    """
    if self.use_ckpt_load_replication:
      self.replication_coordinator = get_replication_coordinator(path)

    if sharded_state_dict is None:
      raise ValueError(
          "DistributedCheckpointIO requires passing sharded_state_dict argument"
          " to load_checkpoint"
      )
    if map_location is not None:
      raise ValueError(
          "DistributedCheckpointIO doesnt handle map_location argument"
      )

    if self.save_ckpt_format == "zarr" and self.load_directly_on_device:
      sharded_strategy = tensorstore.TensorStoreLoadShardedStrategy(
          load_directly_on_device=True
      )
    else:
      sharded_strategy = InClusterLocalCheckpointLoadStrategy(
          self.replication_coordinator
      )

    if self.parallel_load:
      if sharded_strategy is None:
        sharded_strategy = get_default_load_sharded_strategy(path)
      sharded_strategy = FullyParallelLoadStrategyWrapper(
          sharded_strategy, get_data_parallel_group(with_context_parallel=True)
      )

    if sharded_strategy is not None:
      logging.info(f"Using {sharded_strategy} dist-ckpt load strategy.")

    if isinstance(strict, bool):
      # For backward-compatibility reasons and a bug in MCore (strict check not applied to factories)
      # we must apply a simple strict check here.
      if not strict:
        sharded_state_dict = self.adjust_non_strict_load(
            path, sharded_state_dict
        )
      strict = (
          StrictHandling.ASSUME_OK_UNEXPECTED
          if strict
          else StrictHandling.LOG_ALL
      )
    if self.load_strictness is not None:
      # Overwrites function argument
      strict = self.load_strictness
    if strict is None:
      # Default behavior
      strict = StrictHandling.ASSUME_OK_UNEXPECTED

    logging.debug(f"Dist ckpt load strictness: {strict}")

    return in_cluster_load(
        sharded_state_dict=sharded_state_dict,
        checkpoint_dir=path,
        sharded_strategy=sharded_strategy,
        validate_access_integrity=validate_access_integrity,
        strict=strict,
        replication_coordinator=self.replication_coordinator,
    )

  def _determine_dist_ckpt_save_strategy(self):
    """Determine the saving strategy based on constructor args.

    Relies on the default MCore strategy unless extra PyT Distributed format
    arguments
    are passed in config or in case of a fully parallel save in which case
    a parallelization wrapper is applied.
    """
    if self.save_ckpt_format == "zarr":
      logging.warning(
          "`zarr` distributed checkpoint backend is deprecated. Distributed"
          " optimizer checkpoint saving might be extremely slow. Please switch"
          " to PyTorch Distributed format (model.dist_ckpt_format=torch_dist)."
      )

    if self.async_save and self.save_ckpt_format != "torch_dist":
      raise ValueError(
          "Async dist-ckpt save supported only for torch_dist format"
      )

    torch_dist_kwargs = (
        {}
        if self.torch_dist_multiproc is None
        else dict(thread_count=self.torch_dist_multiproc)
    )
    if self.save_ckpt_format == "torch_dist" and torch_dist_kwargs:
      save_strategy = InClusterLocalCheckpointSaveStrategy(
          self.save_ckpt_format, 1, **torch_dist_kwargs
      )
    else:
      save_strategy = get_default_save_sharded_strategy(
          self.save_ckpt_format, 1
      )

    # MCore v0.8 introduces `use_cached_ckpt_structure` attribute
    if hasattr(save_strategy, "use_cached_ckpt_structure"):
      save_strategy.use_cached_ckpt_structure = self.assume_constant_structure

    if self.parallel_save:
      parallelization_group = (
          get_data_parallel_group(with_context_parallel=True)
          if self.parallel_save_within_dp
          else None
      )
      save_strategy = FullyParallelSaveStrategyWrapper(
          save_strategy, parallelization_group, self.assume_constant_structure
      )

    logging.info(f"Using {save_strategy} dist-ckpt save strategy.")
    return save_strategy


class InClusterLocalCheckpointSaveStrategy(TorchDistSaveShardedStrategy):

  def __init__(
      self,
      backend: str,
      version: int,
      keep_only_main_replica: bool = True,
      thread_count: int = 2,
      cached_metadata: bool = False,
      separation_hint: str = None,
  ):
    super().__init__(
        backend,
        version,
        keep_only_main_replica,
        thread_count,
        cached_metadata,
        separation_hint,
    )

    # Required for local checkpoint save implementation
    # TODO: Implement local checkpoint save with `keep_only_main_replica=True`
    self.keep_only_main_replica = False

  def save_common(self, common_state_dict: StateDict, checkpoint_dir: Path):
    if get_is_checkpoint_file_handler(
      is_cluster_local_checkpointing=True,
      is_persistent_storage=False
    ):
      torch.save(common_state_dict, checkpoint_dir / COMMON_STATE_FNAME)

  def async_save(
      self, sharded_state_dict: ShardedStateDict, checkpoint_dir: Path
  ) -> AsyncRequest:
    """Translates MCore ShardedTensors to PyT ShardedTensors & saves in PyT Distributed format.

    Args:
        sharded_state_dict (ShardedStateDict): sharded state dict to save
        checkpoint_dir (Path): checkpoint directory

    Returns: None
    """
    # Translate the state dict
    (sharded_state_dict, flat_mapping, rename_mapping) = (
        _replace_state_dict_keys_with_sharded_keys(
            sharded_state_dict, self.keep_only_main_replica
        )
    )
    pyt_state_dict = mcore_to_pyt_state_dict(sharded_state_dict, False)
    # Use PyT saving mechanism
    writer = InClusterLocalCheckpointFileSystemWriter(
        checkpoint_dir,
        separation_hint=self.separation_hint,
        thread_count=self.thread_count,
    )
    # This should be set differently if we run in a smaller process group than the default
    coordinator = torch.distributed.get_rank()
    # Try twice to validate the generated `central_plan` is the same across iterations
    # If so, reuse `cached_central_plan` and `cached_global_metadata`
    # From the 3rd iteration, `save_state_dict_async_plan` will not generate `global_metadata`
    # (return None) so `self.cached_global_metadata` is reused
    args_cached_plans = None
    if self.use_cached_ckpt_structure:
      args_cached_plans = (
          self.cached_central_plan,
          self.cached_local_plan,
          self.validated_cache_reuse,
      )

    (
        save_state_dict_ret,
        self.cached_central_plan,
        self.cached_local_plan,
        self.validated_cache_reuse,
    ) = save_state_dict_async_plan(
        pyt_state_dict,
        writer,
        None,
        coordinator,
        planner=SkipValidationSavePlanner(
            dedup_replicated_tensors=not self.keep_only_main_replica
        ),
        cached_ckpt_structure=args_cached_plans,
        no_dist=True,
    )
    rank = torch.distributed.get_rank()
    if self.use_cached_ckpt_structure:
      if self.validated_cache_reuse:
        logger.debug(f"rank: {rank}, cache validated")
        if save_state_dict_ret[1]:  # when global_metadata is not cached
          self.cached_global_metadata = save_state_dict_ret[1]  # Cache Metadata
        # Only Coordinator rank holds cached global_metadata
        # (None is returned for global_metadata)
        elif coordinator == rank:
          logger.debug(
              f"rank: {rank}, reuse metadata, {save_state_dict_ret[1]}"
          )
          save_state_dict_ret = list(save_state_dict_ret)
          save_state_dict_ret[1] = self.cached_global_metadata

    return self._get_save_and_finalize_callbacks(writer, save_state_dict_ret)


class InClusterLocalCheckpointLoadStrategy(TorchDistLoadShardedStrategy):
  """Basic load strategy for the PyT Distributed format."""

  def __init__(
      self,
      replication_coordinator: Optional[ReplicationCoordinator] = None,
  ):
    super().__init__()
    self.replication_coordinator = replication_coordinator
    self.metadata = None

  def load(
      self, sharded_state_dict: ShardedStateDict, checkpoint_dir: Path
  ) -> StateDict:
    """Translates MCore ShardedTensors to PyT ShardedTensors & loads from PyT Distributed fmt.

    Args:
        sharded_state_dict (ShardedStateDict): sharded state dict with mapping
          information to instruct loading
        checkpoint_dir (Path): checkpoint directory
        Returns: loaded state dict
    """
    storage_reader = ReplicationStorageReader(
        checkpoint_dir, self.replication_coordinator
    )

    # Read checkpoint metadata and replicate if needed
    self.ckpt_metadata = storage_reader.read_metadata(
        get_local_ckpt_rank(checkpoint_dir)
    )
    if self.replication_coordinator is not None:
      self.ckpt_metadata = self.replication_coordinator.replicate_obj(
          self.ckpt_metadata, broadcast=True
      )

    # Apply N-D tensors resharding
    sharded_state_dict, formulation_restore_data = (
        apply_nd_flattened_tensors_reformulation(
            sharded_state_dict,
            get_reformulation_metadata(
                sharded_state_dict,
                torch.distributed.get_rank(),
                self.ckpt_metadata,
            ),
        )
    )
    flexible_shape_sharded_tensors = [
        sh_ten
        for sh_ten in nested_values(sharded_state_dict)
        if isinstance(sh_ten, ShardedTensor) and not sh_ten.allow_shape_mismatch
    ]

    orig_sharded_state_dict = sharded_state_dict
    # MCore state dict to PyT Distributed compatible
    (sharded_state_dict, flat_mapping, rename_mapping) = (
        _replace_state_dict_keys_with_sharded_keys(sharded_state_dict)
    )
    pyt_state_dict = mcore_to_pyt_state_dict(sharded_state_dict, True)

    # Define planner. If replication coordinator is present, use FlexibleLoadPlanner.
    # This will prevent errors from occuring if local checkpoint is missing.
    if self.replication_coordinator is not None:
      planner = ReplicationLoadPlanner(
          shapes_validation_sharded_tensors=flexible_shape_sharded_tensors,
      )
    else:
      planner = MCoreLoadPlanner(
          shapes_validation_sharded_tensors=flexible_shape_sharded_tensors
      )

    # Load PyT Distributed format
    in_cluster_load_state_dict(
        state_dict=pyt_state_dict,
        storage_reader=storage_reader,
        planner=planner,
        no_dist=True,
        replication_coordinator=self.replication_coordinator,
        local_ckpt_rank=get_local_ckpt_rank(checkpoint_dir),
    )
    pyt_state_dict = cast(
        Dict[str, Union[TorchShardedTensor, List[io.BytesIO]]], pyt_state_dict
    )
    # Unwrap ShardedTensors and return to original state dict
    mcore_state_dict = {}
    for k, v in pyt_state_dict.items():
      if isinstance(v, io.BytesIO):
        # Need to wrap in list to be compatible with broadcast
        mcore_state_dict[k] = [v]
      elif isinstance(v, TorchShardedTensor):
        mcore_state_dict[k] = _unwrap_pyt_sharded_tensor(v)
      else:
        mcore_state_dict[k] = v

    # Broadcast the state dictionary if needed
    if self.replication_coordinator is not None:
      start_broadcast = time()
      if self.replication_coordinator.is_broadcast_participant():
        for k, v in mcore_state_dict.items():
          if isinstance(v, list):
            for i, v_i in enumerate(v):
              mcore_state_dict[k][i] = self.replication_coordinator.broadcast(
                  v_i
              )
          else:
            mcore_state_dict[k] = self.replication_coordinator.broadcast(v)
      end_broadcast = time()
      logger.debug(
          f"[Rank{dist.get_rank()}] Total time to broadcast state dict:"
          f" {end_broadcast - start_broadcast:.2f}s"
      )

    mcore_state_dict = _replace_sharded_keys_with_state_dict_keys(
        mcore_state_dict, flat_mapping, rename_mapping
    )
    _restore_dict_types(mcore_state_dict, orig_sharded_state_dict)
    # Apply N-D tensors resharding postprocessing
    mcore_state_dict = restore_nd_flattened_tensors_formulation(
        mcore_state_dict, formulation_restore_data
    )

    return mcore_state_dict


class InClusterLocalCheckpointFileSystemWriter(FileSystemWriterAsync):

  def __init__(self, *args, separation_hint: Optional[str] = None, **kwargs):
    super().__init__(*args, **kwargs)
    self.separation_hint = separation_hint
    self.rank: Optional[int] = None

  def set_up_storage_writer(
      self, is_coordinator: bool, rank: Optional[int] = None
  ) -> None:
    self.rank = rank

    if self.path:
      path = Path(f"{self.path}/{self.rank}")
      self.path = self.fs.init_path(path)

  def prepare_global_plan(self, plans: List[SavePlan]) -> List[SavePlan]:
    assert len(plans) == 1
    return [
        dataclasses.replace(
            plans[0], storage_data=_StoragePrefix(f"__{self.rank}_")
        )
    ]


def in_cluster_load(
    sharded_state_dict: ShardedStateDict,
    checkpoint_dir: str,
    sharded_strategy: Union[LoadShardedStrategy, Tuple[str, int], None] = None,
    common_strategy: Union[LoadCommonStrategy, Tuple[str, int], None] = None,
    validate_access_integrity: bool = True,
    strict: Union[str, StrictHandling] = StrictHandling.ASSUME_OK_UNEXPECTED,
    replication_coordinator: Optional[ReplicationCoordinator] = None,
) -> Union[StateDict, Tuple[StateDict, Set[str], Set[str]]]:
  """Loading entrypoint.

  In the steps below, the following verbs refer to corresponding objects:
  - load = load from checkpoint
  - extract = extract from sharded_state_dict
  - add = add to the final state dict
  Steps:
  1. Load common state dict and form the base of the result state dict
  2. Apply factories to sharded_state_dict
  3. Extract LocalNonPersistentObject and add
  4. (optional) Extract ShardedObjects, load and add
  5. Extract ShardedBase, load, apply factory merges and add

  Args:
      sharded_state_dict (ShardedStateDict): state dict of the existing model
        populated with ShardedTensors. Used as a mapping to determine which
        parts of global tensors stored in the checkpoint should be loaded.
      checkpoint_dir (str): directory with the checkpoint
      sharded_strategy (LoadShardedStrategy, Tuple[str, int], optional):
        configures loading behavior for sharded tensors
      common_strategy (LoadCommonStrategy, Tuple[str, int], optional):
        configures loading behavior for common data
      validate_access_integrity (bool default = True): checks if each tensor
        shard is accessed exactly once (as main replica) by some process
      strict (StrictHandling, str, optional): determines the behavior in case of
        a mismatch between the requested sharded state dict and the checkpoint.
        See `StrictHandling` docs for more details. Some values affect the
        return value of this function (missing and unexpected keys are
        returned). Defaults to `True` (StrictHandling.ASSUME_OK_UNEXPECTED)
        which doesn't incur any performance overhead. Other recommended values
          are: `False` (StrictHandling.LOG_UNEXPECTED) which logs only
            unexpected keys or `StrictHandling.RETURN_ALL` which returns all
            mismatch keys.
      replication_coordinator (replication_coordinator, optional): coordinator
        for replication operations

  Returns:
      StateDict or Tuple[StateDict, Set[str], Set[str]]: in most cases only
          the loaded state dict is returned. If `strict` flag was set to
  """
  checkpoint_dir = Path(checkpoint_dir)
  common_state_dict = None

  try:
    sharded_strategy, common_strategy = verify_checkpoint_and_load_strategy(
        checkpoint_dir, sharded_strategy, common_strategy
    )
    common_state_dict = common_strategy.load_common(checkpoint_dir)
  except Exception as e:
    # This exception should only occur when the checkpoint_dir does not exist locally,
    # in which case the replication coordinator should exist. Otherwise, raise error
    if replication_coordinator is None:
      raise e

  # Replicate common state dict and common strategy if needed
  if replication_coordinator is not None:
    common_state_dict = replication_coordinator.replicate_obj(
        common_state_dict, broadcast=True
    )
    common_strategy = replication_coordinator.replicate_obj(
        common_strategy, broadcast=True
    )

  sharded_state_dict, nonpersistent_state_dict, sh_ten_factories = (
      load_preprocess(sharded_state_dict)
  )

  merge(common_state_dict, nonpersistent_state_dict)

  # At this point we are only dealing with ShardedBase objects
  sharded_state_dict, _ = extract_sharded_base(sharded_state_dict)

  # Validation
  ckpt_sharded_metadata = None
  local_metadata, global_metadata = None, None
  strict = parse_strict_flag(strict)
  if StrictHandling.requires_explicit_ckpt_mismatch_check(strict):
    ckpt_sharded_metadata = self.load_sharded_metadata(
        str(checkpoint_dir), sharded_strategy, common_strategy
    )
  if validate_access_integrity or StrictHandling.requires_global_app_metadata(
      strict
  ):
    local_metadata, global_metadata = determine_global_metadata(
        sharded_state_dict
    )

  sharded_state_dict, missing_keys, unexpected_keys = (
      validate_integrity_and_strict_load(
          sharded_state_dict,
          strict,
          validate_access_integrity,
          local_metadata,
          global_metadata,
          ckpt_sharded_metadata,
      )
  )

  # ShardedBase loading
  if not sharded_strategy.can_handle_sharded_objects:
    validate_sharded_objects_handling(sharded_strategy, common_strategy)
    sharded_objects_state_dict, sharded_state_dict = extract_matching_values(
        sharded_state_dict, lambda v: isinstance(v, ShardedObject)
    )
    sharded_objects = common_strategy.load_sharded_objects(
        sharded_objects_state_dict, checkpoint_dir
    )
    merge(common_state_dict, sharded_objects)

  loaded_state_dict = sharded_strategy.load(sharded_state_dict, checkpoint_dir)

  merge(common_state_dict, loaded_state_dict)

  loaded_state_dict = apply_factory_merges(common_state_dict, sh_ten_factories)

  if StrictHandling.requires_returning_mismatch_keys(strict):
    return common_state_dict, missing_keys, unexpected_keys
  else:
    return common_state_dict


def in_cluster_load_state_dict(
    state_dict: dict[str, Any],
    storage_reader: StorageReader,
    process_group: Optional[dist.ProcessGroup] = None,
    coordinator_rank: int = 0,
    no_dist: bool = False,
    planner: Optional[LoadPlanner] = None,
    replication_coordinator: Optional[ReplicationCoordinator] = None,
    local_ckpt_rank: Optional[int] = None,
) -> None:
  use_rank_coordination = not no_dist
  distW = _DistWrapper(process_group, not no_dist, coordinator_rank)
  if planner is None:
    planner = DefaultLoadPlanner()

  ckpt_kwargs = {}
  if (ckpt_id := getattr(storage_reader, "checkpoint_id", None)) is not None:
    ckpt_kwargs["checkpoint_id"] = ckpt_id

    @_dcp_method_logger(**ckpt_kwargs)
    def local_step():
      assert planner is not None
      rank = (
          distW.rank if use_rank_coordination else torch.distributed.get_rank()
      )
      if local_ckpt_rank != -1:
        rank = local_ckpt_rank

      # Get local_metadata (and non_local_metadata if replication is used)
      local_metadata = storage_reader.read_metadata(rank=rank)
      if replication_coordinator is not None:
        non_local_metadata = replication_coordinator.replicate_obj(
            local_metadata, broadcast=False
        )
        if not replication_coordinator.is_receiver():
          non_local_metadata = Metadata({})

        planner.set_up_planner(
            state_dict, local_metadata, non_local_metadata, distW.is_coordinator
        )
        storage_reader.set_up_storage_reader(
            local_metadata, non_local_metadata, distW.is_coordinator, rank=rank
        )

      else:
        planner.set_up_planner(state_dict, local_metadata, distW.is_coordinator)
        storage_reader.set_up_storage_reader(
            local_metadata, Metadata({}), distW.is_coordinator, rank=rank
        )

      local_plan = planner.create_local_plan()
      local_plan = storage_reader.prepare_local_plan(local_plan)
      return local_plan

  @_dcp_method_logger(**ckpt_kwargs)
  def global_step(all_local_plans):
    assert planner is not None
    all_local_plans = planner.create_global_plan(all_local_plans)
    all_local_plans = storage_reader.prepare_global_plan(all_local_plans)
    return all_local_plans

  if use_rank_coordination:
    central_plan: LoadPlan = distW.reduce_scatter(
        "plan", local_step, global_step
    )
  else:
    local_plan: LoadPlan = local_step()
    global_plan: LoadPlan = global_step([local_plan])
    central_plan: LoadPlan = global_plan[0]
    torch.distributed.barrier()

  @_dcp_method_logger(**ckpt_kwargs)
  def read_data():
    assert planner is not None
    final_local_plan = planner.finish_plan(central_plan)

    all_reads = storage_reader.read_data(final_local_plan, planner)
    all_reads.wait()
    return None

  start_read = time()
  _ = distW.all_gather("read", read_data)
  end_read = time()
  logger.debug(
      f"[Rank{dist.get_rank()}] Total time to read data (with send/recv if"
      f" applied): {end_read - start_read:.2f}s"
  )


def save_state_dict_async_plan(
    state_dict: STATE_DICT_TYPE,
    storage_writer: "FileSystemWriterAsync",
    process_group: Optional[dist.ProcessGroup] = None,
    coordinator_rank: int = 0,
    planner: Optional[SavePlanner] = None,
    cached_ckpt_structure: Optional[Tuple[SavePlan, SavePlan, bool]] = None,
    no_dist: bool = False,
) -> Tuple[
    Tuple["FileSystemWriterAsync", Metadata, _DistWrapper], SavePlan, bool
]:
  """First stage of saving a state dict to storage.

  This is an async adjustment of torch.distributed.checkpoint.state_dict_saver.
  In order to support async save, saving should be split into three parts:
  1. Planning
  2. Actual saving
  3. Finalization

  Out of these, step (2) *must* happen asynchronously.
  The first step is realized with this function.

  The planning part consists of several steps, described here:
  https://pytorch.org/docs/stable/distributed.checkpoint.html#torch.distributed.checkpoint.SavePlanner

  Args:
      state_dict (STATE_DICT_TYPE): state dict to save
      storage_writer (FileSystemWriterAsync): in current version only an
        instance of FileSystemWriterAsync
      process_group (dist.ProcessGroup, optional): process group used for save
        planning
      coordinator_rank (int, optional): coordinator rank for planning. Defaults
        to 0.
      planner (SavePlanner, optional): save planner for
        torch.distributed.checkpoint format
      cached_ckpt_structure (Tuple[SavePlan, SavePlan, bool], Optional): Each
        object of this tuple will be used in the order as following
          cached_central_plan (SavePlan): a globally coordinated save plan
            cached in the previous iteration
          cached_local_plan (SavePlan): a local plan cached in the previous
            iteration
          validated_cache_reuse (bool): boolean value to tell global_metadata
            and planning dict is consistent over iterations

  Returns: Tuple of:
      - storage writer (the one passed as input)
      - metadata from planning
      - distributed wrapper used for planning
  The return value of this function should be passed as an input to
  `save_state_dict_async_finalize` and cached_plan to skip `reduce_scatter` at
    planning.
  """
  use_rank_coordination = not no_dist
  cached_central_plan, cached_local_plan, validated_cache_reuse = (
      None,
      None,
      False,
  )
  if cached_ckpt_structure:
    cached_central_plan, cached_local_plan, validated_cache_reuse = (
        cached_ckpt_structure
    )

  rank = (
      torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
  )
  dist_wrapper = _DistWrapper(process_group, not no_dist, coordinator_rank)
  if planner is None:
    planner = SkipValidationSavePlanner()
  assert planner is not None

  global_metadata = None
  logger.debug(f"rank: {rank}, starting state dict save")
  local_plan = cached_local_plan

  def local_step():
    nonlocal local_plan
    assert planner is not None
    # PyTorch 2.4 introduced additional `metadata` argument,
    # we have to reference `is_coordinator` args by name
    planner.set_up_planner(
        state_dict, is_coordinator=dist_wrapper.is_coordinator
    )
    rank = (
        dist_wrapper.rank
        if use_rank_coordination
        else torch.distributed.get_rank()
    )
    storage_writer.set_up_storage_writer(dist_wrapper.is_coordinator, rank)
    if not validated_cache_reuse and local_plan is None:
      local_plan = planner.create_local_plan()
    local_plan = storage_writer.prepare_local_plan(local_plan)
    return local_plan

  def global_step(all_local_plans):
    nonlocal global_metadata
    assert planner is not None
    all_local_plans, global_metadata = planner.create_global_plan(
        all_local_plans
    )
    all_local_plans = storage_writer.prepare_global_plan(all_local_plans)
    return all_local_plans

  # Execute local and global planning
  start_plan = time()
  if validated_cache_reuse and cached_central_plan:
    logger.debug(f"rank: {rank}, Passed cache reusable")
    local_step()
    central_plan = cached_central_plan
  else:
    if use_rank_coordination:
      central_plan = dist_wrapper.reduce_scatter(
          "plan", local_step, global_step
      )
    else:
      local_plan: SavePlan = local_step()
      global_plan: SavePlan = global_step([local_plan])
      central_plan: SavePlan = global_plan[0]
      torch.distributed.barrier()
  central_plan = planner.finish_plan(central_plan)
  end_plan = time()
  logger.debug(f"rank: {rank}, plan time: {end_plan - start_plan}")
  # Prepare async writing of tensors.
  # The `storage_writer` will store the information about tensors it needs to save
  start = time()
  storage_writer.prepare_write_data(central_plan, planner)
  end = time()
  logger.debug(f"{time()} rank: {rank}, write(async) time: {end - start}")
  return (
      (storage_writer, cast(Metadata, global_metadata), dist_wrapper),
      central_plan,
      local_plan,
      cached_central_plan == central_plan,
  )


def get_reformulation_metadata(
    sharded_state_dict: ShardedStateDict,
    rank: int,
    ckpt_metadata: Metadata,
) -> Dict[str, TensorReformulationMetadata]:
  """Reads MCore data for N-D flattened tensors from checkpoint metadata during ckpt load.

  Args:
      sharded_state_dict (ShardedStateDict): sharded state dict to load
      rank (int): rank of the process

  Returns:
      Dict[str, TensorReformulationMetadata] - dictionary that maps keys of
      every
          N-D flattened tensor from the sharded_state_dict to its original
          global shape
          as stored in `mcore_data` in the checkpoint.
  """
  reformulation_metadata = {}
  for sh_ten in nested_values(sharded_state_dict):
    if not is_nd_flattened_tensor(sh_ten):
      continue
    try:
      ckpt_global_shape = ckpt_metadata.mcore_data[sh_ten.key][
          "nd_reformulated_orig_global_shape"
      ]
    except KeyError as e:
      raise CheckpointingException(
          "Cannot find global shape metadata for N-D flattened tensor"
          f" {sh_ten} in checkpoint metadata: {ckpt_metadata.mcore_data}"
      ) from e

    reformulation_metadata[sh_ten.key] = TensorReformulationMetadata(
        ckpt_global_shape, ckpt_metadata.state_dict_metadata[sh_ten.key].size
    )
  return reformulation_metadata


class SkipValidationSavePlanner(MCoreSavePlanner):

  def create_global_plan(
      self, all_plans: List[SavePlan]
  ) -> Tuple[List[SavePlan], Metadata]:
    all_plans = dedup_save_plans(all_plans, self.dedup_save_to_lowest_rank)

    global_plan, metadata = create_default_global_save_plan(all_plans)

    if self.flatten_state_dict:
      # | does not work for Python 3.8 or older version.
      # merged_mappings = reduce(
      #     lambda x, y: x | y, (p.planner_data for p in global_plan)
      # )
      planner_data_dict = [p.planner_data for p in global_plan]
      merged_mappings = dict(ChainMap(*planner_data_dict))
      metadata = dataclasses.replace(metadata, planner_data=merged_mappings)

    # if not _validate_global_plan(global_plan, metadata):
    #     raise ValueError("Failed to validate global plan")

    metadata.mcore_data = dict(
        ChainMap(*(plan.mcore_data for plan in all_plans))
    )

    self.global_plan = global_plan
    self.metadata = metadata

    return self.global_plan, self.metadata
