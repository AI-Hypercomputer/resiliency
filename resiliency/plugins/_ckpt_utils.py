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

import os
from pathlib import Path
import re
from typing import Any, Callable, List, Optional, Tuple, Union

from megatron.core.dist_checkpointing.core import CheckpointingConfig, save_config
from megatron.core.dist_checkpointing.dict_utils import extract_matching_values
from megatron.core.dist_checkpointing.mapping import (
    CheckpointingException,
    CommonStateDict,
    ShardedObject,
    ShardedStateDict,
    StateDict,
)
from megatron.core.dist_checkpointing.serialization import (
    get_default_save_common_strategy,
    get_default_save_sharded_strategy,
)
from megatron.core.dist_checkpointing.state_dict_transformation import save_preprocess
from megatron.core.dist_checkpointing.strategies.base import (
    AsyncSaveShardedStrategy,
    SaveCommonStrategy,
    SaveShardedStrategy,
    StrategyAction,
    get_default_strategy,
)
from megatron.core.dist_checkpointing.validation import (
    validate_sharded_objects_handling,
)
from resiliency.third_party.megatron.async_utils import AsyncRequest, PersistentAsyncCaller
from resiliency.third_party.torch._async_process_executor import (
    _AsyncCheckpointProcess,
    _CheckpointRequestIdentifier,
    _CheckpointSaveProcessControlOpts,
    _ProcessGroupInitInfo,
)
from resiliency.third_party.torch.logger import _init_logger
from resiliency.utils import get_resiliency_logger
import torch
from torch import multiprocessing as mp
import torch.distributed as dist
from torch.distributed.checkpoint.metadata import STATE_DICT_TYPE
from torch.distributed.checkpoint.planner import SavePlanner
from torch.distributed.checkpoint.storage import StorageWriter

logger = get_resiliency_logger(__name__)


def find_latest_checkpoint_path(
    checkpoint_dir: Union[str, List[str], Path], synchronize: bool = False
) -> Optional[Path]:
  """Find the latest checkpoint path from a directory or list of directories.

  Args:
    checkpoint_dir: Single directory path or list of directory paths to search
    synchronize: Whether to synchronize across processes

  Returns:
    Path to the latest checkpoint, or None if no checkpoints found
  """
  if isinstance(checkpoint_dir, str) or isinstance(checkpoint_dir, Path):
    checkpoint_dirs = [checkpoint_dir]
  elif isinstance(checkpoint_dir, list):
    checkpoint_dirs = checkpoint_dir
  elif checkpoint_dir is None:
    return None
  else:
    raise TypeError(
        f"Invalid type for checkpoint_dir: {type(checkpoint_dir)}. "
        "Expected str, Path, or List[str|Path]."
    )

  # Determine the maximum step and corresponding path
  max_step, max_step_path = -1, None
  for checkpoint_dir in checkpoint_dirs:
    if checkpoint_dir is None:
      continue
    if not isinstance(checkpoint_dir, Path) and not isinstance(
        checkpoint_dir, str
    ):
      raise TypeError(
          f"Invalid checkpoint directory type: {type(checkpoint_dir)}"
      )
    dist_checkpoints = []
    checkpoint_dir = Path(checkpoint_dir)
    for path in checkpoint_dir.glob("*"):
      if not path.is_dir():
        continue
      if (checkpoint_dir / f"{path.name}-unfinished").exists():
        continue
      if re.match(r"^step=\d+$", path.name):
        step = int(re.search(r"step=(-?\d+)", str(path)).group(1))
        if step > max_step:
          max_step = step
          max_step_path = path
    logger.info(f"{checkpoint_dir} found latest checkpoint path {max_step_path}")

  if synchronize:
    # Synchronize the max step and path across all processes
    if not dist.is_initialized():
      raise CheckpointingException(
          "Distributed process group must be initialized to use"
          " synchronization."
      )

    all_procs_max_steps = [-1] * dist.get_world_size()
    all_procs_max_step_paths = [None] * dist.get_world_size()
    dist.all_gather_object(all_procs_max_steps, max_step)
    dist.all_gather_object(all_procs_max_step_paths, max_step_path)

    # Find the global max step across all processes
    global_max_step = max(all_procs_max_steps)
    if global_max_step > -1:
      # Find the index of the process that has the highest step
      max_step_process_idx = all_procs_max_steps.index(global_max_step)
      # Get the path from that process
      max_step_path = all_procs_max_step_paths[max_step_process_idx]
    else:
      max_step_path = None

  logger.info(f"Picked the lasted ckpt path {max_step_path} to restore.")
  return max_step_path


def get_is_checkpoint_file_handler(
    is_cluster_local_checkpointing: bool = False,
    is_persistent_storage: bool = False,
):
  rank = int(os.getenv("RANK"))
  if is_cluster_local_checkpointing and not is_persistent_storage:
    local_world_size = int(os.getenv("LOCAL_WORLD_SIZE", 8))
    local_rank = int(os.getenv("LOCAL_RANK", rank % local_world_size))
    return local_rank == 0
  return rank == 0


class _MegatronCompatibleAsyncCheckpointProcess(_AsyncCheckpointProcess):
  """Async checkpoint process that calls megatron-based save function"""

  def __init__(
      self,
      pg_init_info: _ProcessGroupInitInfo,
      profile_dir: Optional[str] = None,
  ):
    self.ctx = mp.get_context("spawn")
    self._mp_queue_send: mp.Queue = self.ctx.Queue()
    self._mp_queue_recv: mp.Queue = self.ctx.Queue()

    self._save_process = self.ctx.Process(
        target=self._checkpointing_subprocess,
        args=(
            pg_init_info,
            self._mp_queue_send,
            self._mp_queue_recv,
            profile_dir,
        ),
        daemon=True,
    )

    self._save_process.start()
    response = self._wait_for_response()
    assert response == _CheckpointSaveProcessControlOpts.INIT_COMPLETE

  @staticmethod
  def _execute_save(
      state_dict: STATE_DICT_TYPE,
      checkpoint_request_id: _CheckpointRequestIdentifier,
      sharded_strategy: SaveShardedStrategy,
      storage_writer: Optional[StorageWriter] = None,
      planner: Optional[SavePlanner] = None,
      use_in_cluster_local_ckpts: bool = False,
      storage_options: Optional[Any] = None,
  ) -> None:
    thread_count = state_dict.pop("thread_count")

    save(
        sharded_state_dict=state_dict,
        checkpoint_dir=checkpoint_request_id.checkpoint_id,
        sharded_strategy=sharded_strategy,
        use_in_cluster_local_ckpts=use_in_cluster_local_ckpts,
        is_persistent_storage=storage_options,
    )

  @staticmethod
  def _checkpointing_subprocess(
      pg_init_info: _ProcessGroupInitInfo,
      recv: mp.Queue,
      send: mp.Queue,
      profile_dir: Optional[str] = None,
  ) -> None:
    try:
      _init_logger(pg_init_info.global_rank)

      # Setup environment variables for process group initialization.
      os.environ["TORCHELASTIC_USE_AGENT_STORE"] = "False"
      os.environ["MASTER_ADDR"] = pg_init_info.tcp_store_master_addr
      os.environ["MASTER_PORT"] = str(pg_init_info.tcp_store_master_port)
      os.environ["LOCAL_RANK"] = str(pg_init_info.local_rank)
      os.environ["RANK"] = str(pg_init_info.global_rank)
      os.environ["WORLD_SIZE"] = str(pg_init_info.world_size)
      torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

      logger.info(
          "Initializing dist.ProcessGroup in checkpoint background process"
      )
      dist.init_process_group()

      # Set up debugging logs
      import logging
      from nemo.utils import logging as nemo_logging
      from nemo.utils.mcore_logger import add_handlers_to_mcore_logger

      nemo_logging.setLevel(logging.DEBUG)
      megatron_logger = logging.getLogger("megatron")
      add_handlers_to_mcore_logger()

      # Set log level to DEBUG
      megatron_logger.setLevel(logging.DEBUG)

      # Define a new log format
      rank = dist.get_rank()
      formatter = logging.Formatter(
          f"[{rank}] %(asctime)s - %(name)s - %(levelname)s - %(message)s"
      )

      # Create a console handler with the new format
      console_handler = logging.StreamHandler()
      console_handler.setFormatter(formatter)

      logger.info("Checkpoint background process is running...")
      send.put(_CheckpointSaveProcessControlOpts.INIT_COMPLETE)

      # Serving loop.
      while True:
        logger.info("Waiting for checkpoint save request...")
        obj = recv.get()
        if (
            isinstance(obj, _CheckpointSaveProcessControlOpts)
            and obj == _CheckpointSaveProcessControlOpts.TERMINATE
        ):
          logger.info("Terminating the checkpoint background process.")
          return
        assert isinstance(obj, AsyncRequest)
        logger.info(
            f"Received async checkpoint request with id={obj.call_idx}"  # noqa: G004
        )

        if profile_dir is not None and dist.get_rank() == 0:
          from resiliency.callbacks.profile import MAX_TRACE_ENTRIES
          from viztracer import VizTracer

          staged_state_dict = obj.async_fn_args[0]
          global_step = staged_state_dict["common"]["global_step"]
          tracer = VizTracer(
              tracer_entries=MAX_TRACE_ENTRIES,
              max_stack_depth=50,
              log_torch=True,
          )
          tracer.start()

          logger.info(
              "Started profile in subprocess for checkpointing step"
              f" {global_step}."
          )

        # Call the actual save function
        obj.async_fn(*obj.async_fn_args)

        if profile_dir is not None and dist.get_rank() == 0:
          tracer.stop()
          tracer.save(
              output_file=f"{profile_dir}/ckpt_subproc_rank{dist.get_rank()}_step{global_step}_trace.json"
          )

          logger.info(
              "Finished profile in subprocess for checkpointing step"
              f" {global_step}."
          )

        send.put(obj.call_idx)
        logger.info(
            f"Submitted checkpoint save request for checkpoint_id={obj.call_idx}"  # noqa: G004
        )
    except BaseException as e:
      logger.error(
          f"Checkpoint background process encountered an exception: {e}"  # noqa: G004
      )
      send.put(e)
      raise
    finally:
      logger.info("Checkpoint background process is shutting down...")
      dist.destroy_process_group()


class TorchCompatiblePersistentAsyncCaller(
    _MegatronCompatibleAsyncCheckpointProcess, PersistentAsyncCaller
):

  def __init__(self, profile_dir: Optional[str] = None):
    assert dist.is_initialized(), "Process group must be initialized"
    _MegatronCompatibleAsyncCheckpointProcess.__init__(
        self, _ProcessGroupInitInfo(), profile_dir
    )
    self.queue = self._mp_queue_send
    self.comp_q = self._mp_queue_recv
    self.cur_item = None
    self.cur_idx = -1
    self.process = self._save_process


def save(
    sharded_state_dict: ShardedStateDict,
    checkpoint_dir: str,
    sharded_strategy: Union[SaveShardedStrategy, Tuple[str, int], None] = None,
    common_strategy: Union[SaveCommonStrategy, Tuple[str, int], None] = None,
    validate_access_integrity: bool = False,
    async_sharded_save: bool = False,
    preprocess_common_before_consistancy_check: Callable[
        [CommonStateDict], StateDict
    ] = None,
    use_in_cluster_local_ckpts: bool = False,
    is_persistent_storage: bool = False,
) -> Optional[AsyncRequest]:
  """Saving entrypoint.

  Extracts ShardedTensors from the given state dict. Rank 0 saves the
  "regular" part of the checkpoint to common torch file.
  The ShardedTensors are saved according to a strategy specified by the
  config.

  Steps:
  1. Apply factories
  2. Extract and discard LocalNonPersistentObject
  3. Extract all ShardedBase object
  4. Save all other objects to common.pt
  5. (optional) Extract and save ShardedObjects
  6. Save all ShardedBase objects
  7. Write metadata.json file with backend and version metadata.

  Step (6) can be performed asynchronously (see `async_sharded_save`), in this
  case the actual save is embodied in the returned async request and can be
  scheduled by the external caller. For async request, step (7) is added as
  one of the finalization functions, so that metadata.json is written only
  if the checkpoint is complete.

  Args:
      sharded_state_dict (ShardedStateDict): state dict of the populated with
        ShardedTensors. Used as a mapping to determine how local tensors should
        be saved as global tensors in the checkpoint.
      checkpoint_dir (str): directory to save the checkpoint to
      sharded_strategy (SaveShardedStrategy, Tuple[str, int], optional):
        configures sharded tensors saving behavior and backend
      common_strategy (SaveCommonStrategy, Tuple[str, int], optional):
        configures common data saving behavior and backend
      async_sharded_save (bool, optional): if True, for the sharded state dict
        part an async save implementation will be called, with the AsyncRequest
        being returned to the caller. Note that it is the caller responsibility
        to actually schedule the async save. Defaults to False.
        preprocess_common_before_consistancy_check (Callable[[CommonStateDict],
        StateDict], None): A callable function that will preprocess the common
        state dict (i.e can be used  to remove keys that we expect to be
        different in the state dict). The function must not modify the original
        state dict
      is_persistent_storage (bool, optional): if True, the checkpoint is
        considered to be saved to persistent storage.

  Returns:
      AsyncRequest (optional): if `async_sharded_save` is True, returns
          async request that should be scheduled by the caller of this function.
          None otherwise.
  """
  checkpoint_dir = Path(checkpoint_dir)

  if get_is_checkpoint_file_handler(
      use_in_cluster_local_ckpts, is_persistent_storage
  ):
    if not checkpoint_dir.exists():
      raise CheckpointingException(
          f"Checkpoint destination directory does not exist: {checkpoint_dir}"
          " it should be created before call save()"
      )

  if not use_in_cluster_local_ckpts and get_is_checkpoint_file_handler(
      use_in_cluster_local_ckpts, is_persistent_storage
  ):
    if next(checkpoint_dir.iterdir(), None) is not None:
      raise CheckpointingException(
          f"Checkpoint destination directory ({checkpoint_dir}) is not empty"
      )

  if common_strategy is not None:
    raise NotImplementedError("The only supported common strategy is torch")

  if sharded_strategy is None:
    sharded_strategy = get_default_save_sharded_strategy()
  if not isinstance(sharded_strategy, SaveShardedStrategy):
    assert isinstance(sharded_strategy, tuple), type(sharded_strategy)
    sharded_strategy = get_default_strategy(
        StrategyAction.SAVE_SHARDED, *sharded_strategy
    )

  if common_strategy is None:
    common_strategy = get_default_save_common_strategy()
  if not isinstance(common_strategy, SaveCommonStrategy):
    assert isinstance(common_strategy, tuple), type(common_strategy)
    common_strategy = get_default_strategy(
        StrategyAction.SAVE_COMMON, *common_strategy
    )

  if "common" in sharded_state_dict:
    common_state_dict = sharded_state_dict.pop("common")
  else:
    sharded_state_dict, common_state_dict = save_preprocess(
        sharded_state_dict,
        validate_access_integrity,
        preprocess_common_before_consistancy_check,
    )
  if use_in_cluster_local_ckpts and not is_persistent_storage:
    sharded_strategy.save_common(common_state_dict, checkpoint_dir)
  else:
    common_strategy.save_common(common_state_dict, checkpoint_dir)

  if not sharded_strategy.can_handle_sharded_objects:
    validate_sharded_objects_handling(sharded_strategy, common_strategy)
    sharded_objects_state_dict, sharded_state_dict = extract_matching_values(
        sharded_state_dict, lambda v: isinstance(v, ShardedObject)
    )
    common_strategy.save_sharded_objects(
        sharded_objects_state_dict, checkpoint_dir
    )

  def metadata_finalize_fn():
    if get_is_checkpoint_file_handler(
        use_in_cluster_local_ckpts, is_persistent_storage
    ):
      save_config(
          CheckpointingConfig(
              sharded_strategy.backend, sharded_strategy.version
          ),
          checkpoint_dir,
      )
    dist.barrier()

  if not async_sharded_save:
    sharded_strategy.save(sharded_state_dict, checkpoint_dir)
    metadata_finalize_fn()
    return

  if not isinstance(sharded_strategy, AsyncSaveShardedStrategy):
    raise CheckpointingException(
        f"Cannot apply async_save to non-async strategy {sharded_strategy}"
    )

  async_request = sharded_strategy.async_save(
      sharded_state_dict, checkpoint_dir
  )
  async_request.finalize_fns.append(metadata_finalize_fn)
  return async_request
