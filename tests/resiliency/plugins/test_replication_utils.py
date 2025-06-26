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
import shutil
import socket
import tempfile
from unittest import mock

import pytest
from resiliency.plugins.replication_utils import (
    ReplicationCoordinator,
    get_local_ckpt_rank,
    get_replication_coordinator,
)
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def find_available_port() -> int:
  """Find an available port."""
  sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
  sock.bind(("", 0))
  port = sock.getsockname()[1]
  sock.close()
  return port


def setup_process(
    rank,
    world_size,
    test_func,
    base_temp_dir=None,
    test_case=None,
    backend="nccl",
):
  """Initializes the distributed environment and calls the test function."""

  # Set CUDA device for this process
  device_id = rank % torch.cuda.device_count()
  torch.cuda.set_device(device_id)

  dist.init_process_group(backend=backend, rank=rank, world_size=world_size)

  # Mock out device count to simulate 2 nodes with 4 GPUs each
  with mock.patch("torch.cuda.device_count", return_value=4):
    # Setup mock replica groups: [0, 4], [1, 5], [2, 6], [3, 7]
    replica_group = sorted(
        [rank, (rank + torch.cuda.device_count()) % dist.get_world_size()]
    )
    if test_case is not None:
      test_func(rank, world_size, replica_group, base_temp_dir, test_case)
    else:
      test_func(rank, world_size, replica_group)

  dist.destroy_process_group()


def create_checkpoint_files(
    base_dir, rank_dir_path, rank, checkpoint_rank=None
):
  """Helper function to create checkpoint directory, metadata file, and checkpoint file.

  Args:
    base_dir: Base directory for checkpoints
    rank_dir_path: Path to the node directory (e.g., node0, node1)
    rank: The current process rank
    checkpoint_rank: The rank to create checkpoint for (defaults to current rank
      if None)

  Returns:
    Path: Path object to the created checkpoint directory
  """
  if checkpoint_rank is None:
    checkpoint_rank = rank

  # Create directory structure for this rank
  os.makedirs(os.path.join(rank_dir_path, str(checkpoint_rank)), exist_ok=True)

  # Create metadata file
  with open(
      os.path.join(rank_dir_path, str(checkpoint_rank), ".metadata"), "w"
  ) as f:
    f.write("mock metadata")

  # Create checkpoint file
  with open(
      os.path.join(
          rank_dir_path, str(checkpoint_rank), f"__{checkpoint_rank}_0.distcp"
      ),
      "w",
  ) as f:
    f.write(f"checkpoint data for rank {checkpoint_rank}")

  return Path(rank_dir_path)


def setup_checkpoint_dirs_case1(base_dir, rank):
  """Case 1: Each node has the correct local checkpoints.

  Ranks 0-3 checkpoints are in node0 folder, ranks 4-7 are in node1 folder.
  """
  # Determine which node folder to use
  device_count = torch.cuda.device_count()
  node_folder = "node0" if rank < device_count else "node1"
  rank_dir = os.path.join(base_dir, node_folder)

  # Create checkpoint files for this rank
  return create_checkpoint_files(base_dir, rank_dir, rank)


def setup_checkpoint_dirs_case2(base_dir, rank):
  """Case 2: Each node has the wrong local checkpoints.

  Ranks 0-3 checkpoints are in node1 folder, ranks 4-7 are in node0 folder
  (swapped from case 1).
  """
  device_count = torch.cuda.device_count()
  world_size = dist.get_world_size()

  # Determine which node folder to use
  node_folder = "node0" if rank < device_count else "node1"
  rank_dir = os.path.join(base_dir, node_folder)

  # Create checkpoint files for the different rank
  diff_rank = (rank + device_count) % world_size
  return create_checkpoint_files(
      base_dir, rank_dir, rank, checkpoint_rank=diff_rank
  )


def setup_checkpoint_dirs_case3(base_dir, rank):
  """Case 3: One node has the correct checkpoints, the other does not.

  Node0 has ranks 0-3 with checkpoints 0-3, node1 has ranks 4-7 without.
  """
  device_count = torch.cuda.device_count()
  # Determine which node folder to use
  node_folder = "node0" if rank < device_count else "node1"
  rank_dir = os.path.join(base_dir, node_folder)

  if node_folder == "node0":
    # Ranks 0-3 have checkpoints
    return create_checkpoint_files(base_dir, rank_dir, rank)
  else:
    # No checkpoints for node1
    return Path(rank_dir)


def setup_checkpoint_dirs_case4(base_dir, rank):
  """Case 4: One node has incorrect checkpoints, the other does not have any checkpoints.

  Node0 has no checkpoints, node1 has ranks 0-3 with checkpoints 0-3.
  """
  device_count = torch.cuda.device_count()
  world_size = dist.get_world_size()
  # Determine which node folder to use
  node_folder = "node0" if rank < device_count else "node1"
  rank_dir = os.path.join(base_dir, node_folder)

  if node_folder == "node1":
    # In this case, node1 has checkpoints for ranks 0-3
    diff_rank = rank % device_count  # Maps ranks 4-7 to 0-3
    return create_checkpoint_files(
        base_dir, rank_dir, rank, checkpoint_rank=diff_rank
    )
  else:
    # No checkpoints for node0
    return Path(rank_dir)


def test_mock_get_local_ckpt_rank():
  """Unit test for get_local_ckpt_rank without distributed environment."""
  # Create a temporary directory for testing
  temp_dir = tempfile.mkdtemp()
  try:
    os.makedirs(os.path.join(temp_dir, "0"), exist_ok=True)
    with open(os.path.join(temp_dir, "0", f"__0_0.distcp"), "w") as f:
      f.write(f"checkpoint data for rank 0")

    checkpoint_dir = Path(temp_dir)

    with mock.patch("torch.distributed.get_rank", return_value=0):
      # Test when checkpoints exists locally
      assert get_local_ckpt_rank(checkpoint_dir) == 0

      # Test when checkpoints do not exist locally
      checkpoint_dir = Path("/invalid/path")
      assert get_local_ckpt_rank(checkpoint_dir) == -1

  finally:
    shutil.rmtree(temp_dir)


def run_replication_coordinator_test(
    rank, world_size, replica_group, base_temp_dir, test_case
):
  """Run a single replication coordinator test case.

  This function executes a specific test case to verify the behavior of the
  replication coordinator with different checkpoint directory configurations.

  Args:
    rank: The current process rank
    world_size: The total number of processes
    replica_group: List of ranks in the replica group for this process
    base_temp_dir: Base directory for temporary checkpoint files
    test_case: Which test case to run (1-4)
      1: Every rank has its own checkpoint
      2: Every rank has a different checkpoint
      3: Node0 has checkpoints 0-3, node1 has none
      4: Node1 has checkpoints 0-3, node0 has none
  """
  try:
    if test_case == 1:
      # Test Case 1: Every rank has its own checkpoint
      checkpoint_dir = setup_checkpoint_dirs_case1(
          base_temp_dir + "/case1", rank
      )
      coordinator = get_replication_coordinator(checkpoint_dir, replica_group)

      # Since every rank has its own checkpoint, no replication should be needed
      assert (
          coordinator is None
      ), f"Rank {rank} should not have a coordinator in Case 1"

    elif test_case == 2:
      # Test Case 2: Every rank has a different checkpoint
      checkpoint_dir = setup_checkpoint_dirs_case2(
          base_temp_dir + "/case2", rank
      )
      coordinator = get_replication_coordinator(checkpoint_dir, replica_group)

      # In this case, ranks should coordinate to exchange checkpoints
      assert (
          coordinator is not None
      ), f"Rank {rank} should have a coordinator in Case 2"
      assert (
          coordinator.needs_local_ckpt
      ), f"Rank {rank} should need local checkpoint in Case 2"
      assert (
          coordinator.send_to == coordinator.recv_from
          and coordinator.send_to in replica_group
      ), (
          f"Rank {rank} should be sending and receiving from the same rank in"
          " the expected replica_group in Case 2"
      )

    elif test_case == 3:
      # Test Case 3: Half have checkpoints, half don't
      checkpoint_dir = setup_checkpoint_dirs_case3(
          base_temp_dir + "/case3", rank
      )
      coordinator = get_replication_coordinator(checkpoint_dir, replica_group)

      # Ranks 0-3 should be senders, ranks 4-7 should be receivers or participate in broadcasts
      if rank < 4:
        # First group (has checkpoints)
        assert (
            coordinator is not None
        ), f"Rank {rank} should have a coordinator in Case 3"
        assert (
            not coordinator.needs_local_ckpt
        ), f"Rank {rank} should not need local checkpoint in Case 3"
        assert coordinator.is_broadcast_src() and not coordinator.is_sender(), (
            f"Rank {rank} should be sending and broadcasting source and not"
            " sending in Case 3"
        )
      else:
        # Second group (no checkpoints)
        assert (
            coordinator is not None
        ), f"Rank {rank} should have a coordinator in Case 3"
        assert (
            coordinator.needs_local_ckpt
        ), f"Rank {rank} should need local checkpoint in Case 3"
        assert (
            coordinator.is_broadcast_dest() and not coordinator.is_receiver()
        ), (
            f"Rank {rank} should be broadcast destination and not receiving in"
            " Case 3"
        )

    elif test_case == 4:
      # Test Case 4: Inverse of Case 3 - Node1 has checkpoints for ranks 0-3, Node0 has none
      checkpoint_dir = setup_checkpoint_dirs_case4(
          base_temp_dir + "/case4", rank
      )
      coordinator = get_replication_coordinator(checkpoint_dir, replica_group)

      # Ranks 4-7 should be senders, ranks 0-3 should be receivers or participate in broadcasts
      if rank >= 4:
        # Second group (has checkpoints for ranks 0-3)
        assert (
            coordinator is not None
        ), f"Rank {rank} should have a coordinator in Case 4"
        assert (
            coordinator.needs_local_ckpt
        ), f"Rank {rank} should need local checkpoint in Case 4"
        # First rank in group might be broadcasting to others
        assert (
            coordinator.is_sender()
        ), f"Rank {rank} should be sending in Case 4"
      else:
        # First group (no checkpoints)
        assert (
            coordinator is not None
        ), f"Rank {rank} should have a coordinator in Case 4"
        assert (
            coordinator.needs_local_ckpt
        ), f"Rank {rank} should need local checkpoint in Case 4"
        assert (
            coordinator.is_receiver() or coordinator.is_broadcasting_src()
        ), f"Rank {rank} should be receiving or broadcasting source in Case 4"

  finally:
    # Clean up
    dist.barrier()  # Ensure all processes are done before cleanup
    if rank == 0:
      shutil.rmtree(base_temp_dir)


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is not available"
)
@pytest.mark.parametrize("test_case", [1, 2, 3, 4])
def test_replication_coordinator(test_case):
  """Multiprocessing wrapper for the replication coordinator test.

  This function tests several scenarios on 2 mock nodes with 4 GPUs each.
  Each test case simulates a different checkpoint directory structure and
  verifies
  the behavior of the replication coordinator.

  Args:
    test_case: Which test case to run (1-4)
  """
  world_size = 8

  os.environ["MASTER_ADDR"] = "localhost"
  os.environ["MASTER_PORT"] = str(find_available_port())

  base_temp_dir = tempfile.mkdtemp(prefix="")

  # Run the specific test case
  mp.spawn(
      setup_process,
      args=(
          world_size,
          run_replication_coordinator_test,
          base_temp_dir,
          test_case,
      ),
      nprocs=world_size,
      join=True,
  )


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is not available"
)
def cyclic_send_receive(rank, world_size, _):
  """Test where each rank sends a tensor to the next rank in a cyclic manner."""
  assert world_size == 3, "This test requires exactly 3 processes"

  # Create unique tensor for each rank on CUDA
  tensor = torch.tensor(
      [rank * 10, rank * 10 + 1, rank * 10 + 2],
      dtype=torch.float32,
      device="cuda",
  )

  # Determine send and receive targets
  send_to = (rank + 1) % world_size
  recv_from = (rank - 1) % world_size

  # Create replication coordinator for each rank
  coordinator = ReplicationCoordinator(
      needs_local_ckpt=False,
      send_to=send_to,
      recv_from=recv_from,
      broadcast_group=None,
      broadcast_src=None,
  )

  # Use the coordinator to replicate the tensor
  result = coordinator.replicate_obj(tensor, broadcast=False)

  # Validate result - each rank should have received tensor from previous rank
  expected_tensor = torch.tensor(
      [recv_from * 10, recv_from * 10 + 1, recv_from * 10 + 2],
      dtype=torch.float32,
      device="cuda",
  )

  assert torch.all(
      torch.eq(result, expected_tensor)
  ), f"Rank {rank} expected {expected_tensor} but got {result}"


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is not available"
)
def test_cyclic_replication():
  """Multiprocessing wrapper for the cyclic send/receive test."""
  world_size = 3
  mp.spawn(
      setup_process,
      args=(world_size, cyclic_send_receive),
      nprocs=world_size,
      join=True,
  )


if __name__ == "__main__":
  # Allow running the tests directly via python
  test_mock_get_local_ckpt_rank()
  test_cyclic_replication()
  for test_case in range(1, 5):
    test_replication_coordinator(test_case)
