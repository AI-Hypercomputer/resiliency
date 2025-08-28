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

import atexit
import copy
from dataclasses import dataclass
import io
import os
from pathlib import Path
import pickle
import time
from typing import Any, Dict, IO, List, Optional, Union, cast
from lightning.fabric.plugins import ClusterEnvironment
from lightning.fabric.utilities.rank_zero import rank_zero_info
from lightning.fabric.utilities.seed import reset_seed
from lightning.pytorch.strategies.ddp import DDPStrategy
from megatron.core import parallel_state
from megatron.core.dist_checkpointing.core import CheckpointingException
from megatron.core.dist_checkpointing.strategies.torch import MCoreLoadPlanner
from nemo import lightning as nl
from nemo.lightning.pytorch.strategies.utils import init_model_parallel, setup_parallel_ranks
from resiliency.utils import get_resiliency_logger
import torch
from torch import Tensor
import torch.distributed as dist
from torch.distributed._shard._utils import narrow_tensor_by_index
from torch.distributed.checkpoint import (
    BytesStorageMetadata,
    FileSystemReader,
    LoadPlan,
    Metadata,
    ReadItem,
)
from torch.distributed.checkpoint._nested_dict import flatten_state_dict
from torch.distributed.checkpoint._sharded_tensor_utils import _flatten_sharded_tensors
from torch.distributed.checkpoint.default_planner import create_default_local_load_plan
from torch.distributed.checkpoint.filesystem import _StorageInfo
from torch.distributed.checkpoint.metadata import (
    Metadata,
    MetadataIndex,
    STATE_DICT_TYPE,
)
from torch.distributed.checkpoint.planner import LoadItemType
from torch.distributed.checkpoint.planner_helpers import (
    _create_read_item_for_byteio,
    _init_state_dict,
    create_read_items_for_chunk_list,
)
from torch.futures import Future
from tqdm import tqdm

logger = get_resiliency_logger(__name__)


class ReplicationCoordinator:
  """Coordinator for replication operations."""

  def __init__(
      self,
      needs_local_ckpt: bool,
      send_to: int = None,
      recv_from: int = None,
      broadcast_group: dist.ProcessGroup = None,
      broadcast_src: int = None,
  ):
    self.needs_local_ckpt = needs_local_ckpt
    self.send_to = send_to
    self.recv_from = recv_from
    self.broadcast_group = broadcast_group
    self.broadcast_src = broadcast_src

  def is_sender(self) -> bool:
    """Determines whether this ranks should send data to another rank.

    Returns: True if this rank should send data to another rank, False
    otherwise.
    """
    return self.send_to is not None

  def is_receiver(self) -> bool:
    """Determines whether this ranks should receive data from another rank.

    Returns:
        bool: True if this rank should receive data from another rank, False
        otherwise.
    """
    return self.recv_from is not None

  def is_broadcast_src(self) -> bool:
    """Determines whether this rank should be the source of a broadcast operation.

    Returns: True if this rank should be the source of a broadcast operation,
    False otherwise.
    """
    return (
        self.is_broadcast_participant()
        and dist.get_rank() == self.broadcast_src
    )

  def is_broadcast_dest(self) -> bool:
    """Determines whether this rank should be the destination of a broadcast operation.

    Returns: True if this rank should be the destination of a broadcast
    operation, False otherwise.
    """
    return (
        self.is_broadcast_participant()
        and dist.get_rank() != self.broadcast_src
    )

  def is_broadcast_participant(self) -> bool:
    """Determines whether this rank should participate in a broadcast operation.

    Returns: True if this rank should participate in a broadcast operation,
    False otherwise.
    """
    return self.broadcast_group is not None

  def replicate_obj(self, obj: Any, broadcast=True) -> Any:
    """Replicates provided object.

    This class ensures no deadlocks when replicating object.

    Args:
        obj (Any): Object send/recv/broadcast with peers.
        broadcast (bool, optional): Whether to broadcast the object if needed.
          Defaults to True.

    Returns:
        Any: If this rank receives an object via recv or broadcast operation,
        the received object is returned.
        Otherwise, the original object is returned.
    """
    if isinstance(obj, Tensor):
      recv_obj = torch.empty(obj.shape, device=obj.device, dtype=obj.dtype)
    else:
      recv_obj = None

    if self.is_sender() and self.is_receiver():
      same_peer = self.send_to == self.recv_from
      if self.send_to < self.recv_from or (
          same_peer and dist.get_rank() < self.send_to
      ):
        self.send(obj)
        recv_obj = self.recv(recv_obj)
        obj = recv_obj
      else:
        recv_obj = self.recv(recv_obj)
        self.send(obj)
        obj = recv_obj
    elif self.is_sender():
      self.send(obj)
    elif self.is_receiver():
      recv_obj = self.recv(recv_obj)
      obj = recv_obj
    else:
      assert (
          self.is_broadcast_participant()
      ), "`replicate_obj` called with no operation to perform."

    if broadcast and self.is_broadcast_participant():
      obj = self.broadcast(obj)

    return obj

  def send(self, obj: Any, return_to_orig_device: bool = True) -> None:
    """Send object to another rank."""
    if isinstance(obj, torch.Tensor):
      moved = False
      if obj.is_cpu:
        moved = True
        obj = obj.cuda()
      dist.send(obj, dst=self.send_to)
      if return_to_orig_device and moved:
        obj = obj.cpu()
    else:
      obj = [obj]
      dist.send_object_list(obj, dst=self.send_to)

  def recv(self, obj: Any, return_to_orig_device: bool = True) -> Any:
    """Receive object from another rank."""
    if isinstance(obj, torch.Tensor):
      moved = False
      if obj.is_cpu:
        moved = True
        obj = obj.cuda()
      dist.recv(obj, src=self.recv_from)
      if return_to_orig_device and moved:
        obj = obj.cpu()
    else:
      obj = [obj]
      dist.recv_object_list(obj, src=self.recv_from)
      obj = obj[0]
    return obj

  def broadcast(self, obj: Any, return_to_orig_device: bool = True) -> None:
    """Broadcast object to a group of ranks."""
    if isinstance(obj, torch.Tensor):
      moved = False
      if obj.is_cpu:
        moved = True
        obj = obj.cuda()
      dist.broadcast(obj, src=self.broadcast_src, group=self.broadcast_group)
      if return_to_orig_device and moved:
        obj = obj.cpu()
    else:
      obj = [obj]
      dist.broadcast_object_list(
          obj, src=self.broadcast_src, group=self.broadcast_group
      )
      obj = obj[0]
    return obj


@dataclass(frozen=True)
class ReplicationReadItem(ReadItem):
  is_local: bool


class ReplicationLoadPlanner(MCoreLoadPlanner):

  def set_up_planner(
      self,
      state_dict: STATE_DICT_TYPE,
      local_metadata: Optional[Metadata] = None,
      non_local_metadata: Optional[Metadata] = None,
      is_coordinator: bool = False,
  ) -> None:
    _init_state_dict(state_dict)
    self.original_state_dict = state_dict

    if self.flatten_sharded_tensors:
      state_dict = _flatten_sharded_tensors(state_dict)

    if self.flatten_state_dict:
      state_dict, self.mappings = flatten_state_dict(state_dict)

    self.state_dict = state_dict
    self.local_metadata = local_metadata
    self.non_local_metadata = non_local_metadata

    # Define self.metadata to be compatible with MCoreLoadPlanner
    if self.non_local_metadata is not None:
      self.metadata = self.non_local_metadata
    elif self.local_metadata is not None:
      self.metadata = self.local_metadata
    else:
      self.metadata = None
    self.is_coordinator = is_coordinator

  def create_local_plan(self) -> list[LoadPlan]:
    # Create two separate plans, then merge them together
    # with attribute to indicate whether the item is local or not.
    local_plan = create_local_load_plan(self.local_metadata)
    non_local_plan = create_default_local_load_plan(
        self.state_dict, self.non_local_metadata, False
    )

    load_plan = LoadPlan([])
    for item in local_plan.items:
      item = ReplicationReadItem(
          type=item.type,
          dest_index=item.dest_index,
          dest_offsets=item.dest_offsets,
          storage_index=item.storage_index,
          storage_offsets=item.storage_offsets,
          lengths=item.lengths,
          is_local=True,
      )
      load_plan.items.append(item)

    for item in non_local_plan.items:
      item = ReplicationReadItem(
          type=item.type,
          dest_index=item.dest_index,
          dest_offsets=item.dest_offsets,
          storage_index=item.storage_index,
          storage_offsets=item.storage_offsets,
          lengths=item.lengths,
          is_local=False,
      )
      load_plan.items.append(item)
    return load_plan


class ReplicationStorageReader(FileSystemReader):

  def __init__(
      self,
      path: Union[str, os.PathLike],
      replication_coordinator: ReplicationCoordinator,
  ) -> None:
    super().__init__(path)
    self.local_storage_data: Dict[MetadataIndex, _StorageInfo] = {}
    self.non_local_storage_data: Dict[MetadataIndex, _StorageInfo] = {}
    self.replication_coordinator = replication_coordinator

  def reset(self, checkpoint_id: Union[str, os.PathLike, None] = None) -> None:
    super().reset(checkpoint_id)
    self.local_storage_data = {}
    self.non_local_storage_data = {}

  def set_up_storage_reader(
      self,
      local_metadata: Metadata,
      non_local_metadata: Metadata,
      is_coordinator: bool = True,
      rank: Optional[int] = None,
  ) -> None:
    self.rank = rank

    if self.path:
      self.path = Path(f"{self.path}")

    self.local_storage_data = local_metadata.storage_data
    self.non_local_storage_data = non_local_metadata.storage_data

  def read_metadata(self, rank: Optional[int] = None) -> Metadata:
    self.rank = rank

    path = self.fs.concat_path(self.path, f"{rank}/.metadata")
    path_exists = self.fs.exists(path)
    if self.replication_coordinator is not None:
      metadata = Metadata({})

      if path_exists:
        with self.fs.create_stream(path, "rb") as metadata_file:
          metadata = pickle.load(metadata_file)
    else:
      with self.fs.create_stream(path, "rb") as metadata_file:
        metadata = pickle.load(metadata_file)

    return metadata

  def read_data(
      self, plan: LoadPlan, planner: MCoreLoadPlanner
  ) -> Future[None]:
    """Reads data based on the provided load plan and handles replication if enabled.

    This method processes the load plan by organizing read items by file,
    iterating through
    the files, and performing the following operations:
    1. If the file exists locally:
       - Reads the file and processes its content.
       - Sends the data to a replication coordinator if replication is enabled.
    2. If the file does not exist locally:
       - Receives the data from a replication coordinator if replication is
       enabled.
       - Raises a FileNotFoundError if replication is not enabled.

    The method supports two types of load items:
    - BYTE_IO: Reads raw bytes from the file.
    - Tensor: Loads tensors from the file and optionally narrows them by index.

    Args:
        plan (LoadPlan): The plan containing the items to be loaded.
        planner (ReplicationLoadPlanner): The planner responsible for managing
          the loading process.

    Raises:
        FileNotFoundError: If a file does not exist locally and replication is
        not enabled.

    Returns:
        Future[None]: A future that resolves when the data loading process is
        complete.
    """
    total_time_reading_files = 0
    total_time_sending_data = 0
    total_time_receiving_data = 0

    per_file: Dict[str, List[ReadItem]] = {}

    # Organize read items by file
    for read_item in plan.items:
      if not hasattr(read_item, "is_local"):
        item_md = self.local_storage_data[read_item.storage_index]
      elif read_item.is_local:
        item_md = self.local_storage_data[read_item.storage_index]
      else:
        item_md = self.non_local_storage_data[read_item.storage_index]
      path = f"{self.rank}/" + item_md.relative_path
      per_file.setdefault(path, []).append(read_item)

    # Sort per_file dictionary so that ranks that need to coordinate are in sync
    per_file = dict(sorted(per_file.items()))

    for relative_path, reqs in per_file.items():
      new_path = self.fs.concat_path(self.path, relative_path)
      if self.fs.exists(new_path):
        with self.fs.create_stream(new_path, "rb") as stream:
          # TODO sort by offset and cache the reading
          for req in reqs:
            item_md = self.local_storage_data[req.storage_index]

            file_slice = self._slice_file(stream, item_md)
            if req.type == LoadItemType.BYTE_IO:
              # Read object from file
              s = time.time()
              read_bytes = io.BytesIO(file_slice.read(item_md.length))
              total_time_reading_files += time.time() - s
              read_bytes.seek(0)

              if self.replication_coordinator is not None:
                # Send data if needed
                if self.replication_coordinator.is_sender():
                  self.replication_coordinator.send(read_bytes)
                # Skip loading if this rank still needs to receive data
                if (
                    self.replication_coordinator.is_receiver()
                    or self.replication_coordinator.is_broadcast_dest()
                ):
                  continue
              planner.load_bytes(req, read_bytes)
            else:
              s = time.time()
              tensor = cast(
                  Tensor,
                  torch.load(
                      cast(IO[bytes], file_slice),
                      map_location="cpu",
                      weights_only=True,
                  ),
              )
              total_time_reading_files += time.time() - s
              tensor = narrow_tensor_by_index(
                  tensor, req.storage_offsets, req.lengths
              )

              if self.replication_coordinator is not None:
                # Send data if needed. The tensor properties are sent first
                # so that the receiver can pre-allocate the tensor.
                if self.replication_coordinator.is_sender():
                  tensor_properties = (
                      tensor.shape,
                      tensor.dtype,
                      tensor.device,
                  )
                  s = time.time()
                  self.replication_coordinator.send(tensor_properties)
                  self.replication_coordinator.send(
                      tensor, return_to_orig_device=False
                  )
                  total_time_sending_data += time.time() - s
                # Skip loading if this rank still needs to receive data
                if (
                    self.replication_coordinator.is_receiver()
                    or self.replication_coordinator.is_broadcast_dest()
                ):
                  continue

              target_tensor = planner.resolve_tensor(req).detach()

              assert target_tensor.size() == tensor.size(), (
                  f"req {req.storage_index} mismatch sizes"
                  f" {target_tensor.size()} vs {tensor.size()}"
              )
              target_tensor.copy_(tensor)
              planner.commit_tensor(req, target_tensor)
      elif self.replication_coordinator is not None:
        # For every request, this rank will need to receive data before being able to laod it.
        # This is a no-op if rank does not have any data to receive.
        for req in reqs:
          if req.type == LoadItemType.BYTE_IO:
            if self.replication_coordinator.is_receiver():
              s = time.time()
              read_bytes = self.replication_coordinator.recv(None)
              total_time_receiving_data += time.time() - s
              planner.load_bytes(req, read_bytes)
          else:
            if self.replication_coordinator.is_receiver():
              # Receive tensor properties first to pre-allocate tensor.
              s = time.time()
              tensor_properties = self.replication_coordinator.recv(
                  (None, None, None)
              )
              total_time_receiving_data += time.time() - s
              tensor = torch.empty(
                  tensor_properties[0],
                  dtype=tensor_properties[1],
                  device=tensor_properties[2],
              )
              s = time.time()
              tensor = self.replication_coordinator.recv(
                  tensor, return_to_orig_device=False
              )
              total_time_receiving_data += time.time() - s
              target_tensor = planner.resolve_tensor(req).detach()

              assert target_tensor.size() == tensor.size(), (
                  f"req {req.storage_index} mismatch sizes"
                  f" {target_tensor.size()} vs {tensor.size()}"
              )
              target_tensor.copy_(tensor)
              planner.commit_tensor(req, target_tensor)
      else:
        raise FileNotFoundError(
            f"File {new_path} not found. Enable replication."
        )

    fut: Future = Future()
    fut.set_result(None)

    logger.debug(
        f"[Rank{dist.get_rank()}] Total time spent reading files:"
        f" {total_time_reading_files:.2f}s"
    )
    logger.debug(
        f"[Rank{dist.get_rank()}] Total time spent sending data:"
        f" {total_time_sending_data:.2f}s"
    )
    logger.debug(
        f"[Rank{dist.get_rank()}] Total time spent receiving data:"
        f" {total_time_receiving_data:.2f}s"
    )

    return fut


def create_local_load_plan(metadata) -> LoadPlan:
  requests = []
  for fqn, md in metadata.state_dict_metadata.items():
    if not isinstance(md, BytesStorageMetadata):
      try:
        local_chunks = md.chunks
      except ValueError as ex:
        raise CheckpointingException(
            f"Invalid checkpoint metadata for {fqn}, "
            + f"expected BytesStorageMetadata but found {type(md)}",
        ) from ex
      requests += create_read_items_for_chunk_list(fqn, md, local_chunks)
    else:
      requests += [
          _create_read_item_for_byteio(
              dest_index=MetadataIndex(fqn),
              dest_offset=0,
              storage_index=MetadataIndex(fqn),
              storage_offset=0,
              length=0,
          )
      ]

  return LoadPlan(requests)


def get_local_ckpt_rank(checkpoint_dir: Path) -> int:
  """Determines the rank of the local checkpoint to load.

  Args:
      checkpoint_dir (Path): Path to the checkpoint directory.

  Returns:
      int: Rank of the local checkpoint.
  """
  local_ckpt_rank = -1
  if checkpoint_dir.exists():
    all_local_ckpt_ranks = sorted([
        int(loc.name) for loc in checkpoint_dir.iterdir() if loc.name.isdigit()
    ])
    if dist.get_rank() in all_local_ckpt_ranks:
      # Use local checkpoint matching current rank
      local_ckpt_rank = dist.get_rank()
    else:
      # Find local checkpoint compatible to load
      num_devices = torch.cuda.device_count()
      for rank in all_local_ckpt_ranks:
        # This assumes that each node has the same number of training processes
        if dist.get_rank() % num_devices == rank % num_devices:
          if local_ckpt_rank != -1:
            raise CheckpointingException(
                "Multiple local checkpoints found. Unable to determine which"
                " one to load."
            )
          local_ckpt_rank = rank

    if local_ckpt_rank == -1:
      raise CheckpointingException(
          "Unable to determine rank of local checkpoint."
      )

  return local_ckpt_rank


def get_replica_peers(replica_group: List[int] = None) -> Dict[int, List[int]]:
  """Returns a dict mapping each global rank to a list of its replica peers.

  Args:
      replica_group (List[int], optional): List of ranks in the replica group.
        Defaults to None.
  """

  # Gather all rank info
  if replica_group is None:
    replica_group = dist.get_process_group_ranks(
        parallel_state.get_inter_partial_data_parallel_group()
    )
  all_replica_groups = [None] * dist.get_world_size()
  dist.all_gather_object(all_replica_groups, replica_group)

  # Create a mapping of ranks to their replica peers
  replica_groups = {}
  for replica_group in all_replica_groups:
    for rank in replica_group:
      replica_groups[rank] = replica_group
  return replica_groups


def get_replication_coordinator(
    checkpoint_dir: Path,
    replica_group: List[int] = None,
) -> Optional[ReplicationCoordinator]:
  """Determines if current rank should participate in checkpoint replication and returns the corresponding replication coordinator.

  This function has the following algorithm (TODO: Performance improvements):
  1) Collect process rank
  2) Collect local checkpoint rank (-1 if no local checkpoint)
  3) (1) and (2) is collected for all ranks.
  4) Determine mapping of ranks <-> local checkpoint rank
  5) Determine all other ranks have replicated state of this rank.
  6) With this information, we iterate throguh all ranks again. For each rank,
  calculate:
      a) if rank already has access to its own checkpoint: this rank does not
      need to receive.
      b) if rank does not have access to its own checkpoint: this rank needs to
      receive checkpoint from peer rank.
      Peer rank needs to send checkpoint to this rank
          i) if this rank's checkpoint exists somewhere (using (4)), determine
          that rank with checkpoint as source
          ii) if this rank's checkpoint does not exist somewhere (using (4)),
          determine smallest rank with repliated state as source
              smallest peer rank with replicated state will be source of
              broadcast to all ranks that need replicated state
              NOTE: if no replicated state has source, then we cannot rely on
              local checkpointing.
              TODO: Revert to previous checkpoint if local checkpoint does not
              work
  7) Create broadcast groups
  8) Create replication coordinator objects with all of the above information

  Args:
      checkpoint_dir (Path): Path to the checkpoint directory.
      replica_group (List[int], optional): List of ranks in the replica group.
        Replica group is defined as all ranks that have the same weights. For
        example if you have two data replicas, one include ranks 0-3 and the
        other include ranks 4-7, then the replica groups are [0, 4], [1, 5], [2,
        6], [3, 7].

  Raises:
      CheckpointException: An error is raised if at least one rank needs state,
      but no
      other rank in its data parallel group has state.

  Returns:
      Optional[ReplicationCoordinator]: ReplicationCoordinator object if current
      rank should participate in replication, None otherwise.
  """

  logger.info("Starting replication coordinator setup.")
  # Step 1
  global_rank = dist.get_rank()
  world_size = dist.get_world_size()

  # Step 2
  local_ckpt_rank = get_local_ckpt_rank(checkpoint_dir)

  # Step 3
  avail_ckpt_info = torch.tensor([global_rank, local_ckpt_rank], device="cuda")
  all_avail_ckpt_info = [
      torch.tensor([0, 0], device="cuda") for _ in range(world_size)
  ]
  dist.all_gather(all_avail_ckpt_info, avail_ckpt_info)

  # Step 4
  ckpt_to_process_map = {
      tensor[1].item(): tensor[0].item()
      for tensor in all_avail_ckpt_info
      if tensor[1].item() != -1
  }
  process_to_ckpt_map = {v: k for k, v in ckpt_to_process_map.items()}

  # Sort mappings for convenience
  ckpt_to_process_map = dict(sorted(ckpt_to_process_map.items()))
  process_to_ckpt_map = dict(sorted(process_to_ckpt_map.items()))

  # Step 5
  replica_peers = get_replica_peers(replica_group)

  logger.info(f"ckpt_to_process_map: {ckpt_to_process_map}")
  logger.info(f"process_to_ckpt_map: {process_to_ckpt_map}")
  logger.info(f"replica_peers: {replica_peers}")

  # Track all needed communications
  empty_comms = {
      "send": None,
      "recv": None,
      "broadcast": set(),
      "broadcast_src": None,
  }
  all_comms = [copy.deepcopy(empty_comms) for _ in range(world_size)]

  # Step 6
  for rank in range(world_size):
    if rank in ckpt_to_process_map and rank == ckpt_to_process_map[rank]:

      # Step 6a
      continue
    else:

      # Step 6b
      if rank in ckpt_to_process_map:

        # Step 6bi
        all_comms[rank]["recv"] = ckpt_to_process_map[rank]
        all_comms[ckpt_to_process_map[rank]]["send"] = rank
      else:

        # Step 6bii
        replica_group = sorted(replica_peers[rank])
        found_peer = False
        for peer_rank in replica_group:
          if peer_rank in ckpt_to_process_map:
            found_peer = True
            all_comms[peer_rank]["broadcast"].add(peer_rank)
            all_comms[peer_rank]["broadcast"].add(rank)
            all_comms[peer_rank]["broadcast_src"] = peer_rank
            for broadcast_group_rank in all_comms[peer_rank]["broadcast"]:
              all_comms[broadcast_group_rank]["broadcast"] = all_comms[
                  peer_rank
              ]["broadcast"]
              all_comms[broadcast_group_rank]["broadcast_src"] = peer_rank
            break

        if not found_peer:
          raise CheckpointingException(
              "Need to revert to an older checkpoint due to missing state. "
          )

  # Step 7
  broadcast_group = None
  for comms in all_comms:
    # Sort and convert to list for cleanliness
    comms["broadcast"] = sorted(list(comms["broadcast"]))

    if comms["broadcast"]:
      group = dist.new_group(comms["broadcast"])
      if global_rank in comms["broadcast"]:
        broadcast_group = group

  # Log replication information
  for rank, comms in enumerate(all_comms):
    logger.info(f"Rank {rank} replication status:")
    logger.info(f"     Sending to {comms['send']}")
    logger.info(f"     Receiving from {comms['recv']}")
    logger.info(f"     Broadcasting with {comms['broadcast']}.")
    logger.info(f"     Broadcast src rank: {comms['broadcast_src']}")

  # Check if this rank needs to participate in replication
  empty_comms["broadcast"] = []
  if all_comms[global_rank] == empty_comms:
    return None

  # Step 8
  return ReplicationCoordinator(
      needs_local_ckpt=not global_rank == local_ckpt_rank,
      send_to=all_comms[global_rank]["send"],
      recv_from=all_comms[global_rank]["recv"],
      broadcast_group=broadcast_group,
      broadcast_src=all_comms[global_rank]["broadcast_src"],
  )


class ReplicatedOptimizerMegatronStrategy(nl.MegatronStrategy):

  def setup_distributed(self) -> None:
    """Setups dist env"""
    setup_parallel_ranks(self)
    super(nl.MegatronStrategy, self).setup_distributed()
    self._init_model_parallel_with_replicated_optimizer(self.model)

    if self.data_sampler:
      assert isinstance(
          self.cluster_environment, ClusterEnvironment
      ), "Cluster environment not initialized"
      self.data_sampler.setup(self.cluster_environment.global_rank())

  def _init_model_parallel_with_replicated_optimizer(self, model):
    """Initializes Megatron-LM model parallel if using model parallelism."""
    import torch.distributed
    from megatron.core import parallel_state

    from nemo.utils import AppState

    app_state = AppState()

    # we initialize megatron-lm model parallel and data parallel groups
    # after initializing DDP with PTL.
    if app_state.model_parallel_size is not None:
      # destroy groups in case they have already been created
      # this happens with multiple calls to trainer.test for example
      parallel_state.destroy_model_parallel()
      if torch.distributed.is_initialized():
        parallel_state.initialize_model_parallel(
            tensor_model_parallel_size=app_state.tensor_model_parallel_size,
            pipeline_model_parallel_size=app_state.pipeline_model_parallel_size,
            virtual_pipeline_model_parallel_size=app_state.virtual_pipeline_model_parallel_size,
            pipeline_model_parallel_split_rank=app_state.pipeline_model_parallel_split_rank,
            encoder_pipeline_model_parallel_size=app_state.encoder_pipeline_model_parallel_size,
            encoder_tensor_model_parallel_size=app_state.encoder_tensor_model_parallel_size,
            context_parallel_size=app_state.context_parallel_size,
            expert_model_parallel_size=app_state.expert_model_parallel_size,
            num_distributed_optimizer_instances=self.ddp_config.num_distributed_optimizer_instances,
        )

        # assert that fake tp and pp rank match after model parallel init
        assert (
            app_state.tensor_model_parallel_rank
            == parallel_state.get_tensor_model_parallel_rank()
        )
        assert (
            app_state.pipeline_model_parallel_rank
            == parallel_state.get_pipeline_model_parallel_rank()
        )

        app_state.tensor_model_parallel_group = (
            parallel_state.get_tensor_model_parallel_group()
        )
        app_state.data_parallel_group = parallel_state.get_data_parallel_group()
        app_state.data_parallel_rank = parallel_state.get_data_parallel_rank()
        app_state.data_parallel_size = (
            parallel_state.get_data_parallel_world_size()
        )
        app_state.pipeline_model_parallel_group = (
            parallel_state.get_pipeline_model_parallel_group()
        )

        # create MPI process group for UCX-based communication APIs
        if app_state.init_mpi_proc_group:
          torch.distributed.new_group(backend="mpi")
