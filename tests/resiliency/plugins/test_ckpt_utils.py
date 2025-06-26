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

import pytest
from resiliency.plugins._ckpt_utils import find_latest_checkpoint_path
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
    rank, world_size, test_func, temp_dirs=None, test_case=None, backend="nccl"
):
  """Initializes the distributed environment and calls the test function."""
  # Set CUDA device for this process
  device_id = rank % torch.cuda.device_count()
  torch.cuda.set_device(device_id)

  dist.init_process_group(backend=backend, rank=rank, world_size=world_size)

  if test_case is not None:
    test_func(rank, world_size, temp_dirs, test_case)
  else:
    test_func(rank, world_size, temp_dirs)

  dist.destroy_process_group()


def create_checkpoint_dir(base_dir, step_number, create_unfinished=False):
  """Helper to create a checkpoint directory with the step=X format.

  Args:
    base_dir: Base directory to create checkpoint in
    step_number: Step number for the checkpoint
    create_unfinished: If True, creates an unfinished marker file

  Returns:
    Path to the created checkpoint directory
  """
  checkpoint_dir = Path(base_dir) / f"step={step_number}"
  checkpoint_dir.mkdir(parents=True, exist_ok=True)

  # Create a dummy file to make it a valid checkpoint
  (checkpoint_dir / "dummy_checkpoint.pt").touch()

  if create_unfinished:
    unfinished_marker = Path(base_dir) / f"step={step_number}-unfinished"
    unfinished_marker.touch()

  return checkpoint_dir


def test_find_latest_checkpoint_input_validation():
  """Unit test for find_latest_checkpoint_path input validation."""
  # Test with None input
  result = find_latest_checkpoint_path(None)
  assert result is None, "Should return None for None input"

  # Test with empty string
  result = find_latest_checkpoint_path("")
  assert result is None, "Should return None for empty string input"

  # Test with empty Path object
  result = find_latest_checkpoint_path(Path(""))
  assert result is None, "Should return None for empty Path input"

  # Test with non-string input
  with pytest.raises(TypeError):
    find_latest_checkpoint_path(123)  # Should raise TypeError


def test_find_latest_checkpoint_single_dir():
  """Unit test for find_latest_checkpoint_path with single directory."""
  temp_dir = tempfile.mkdtemp()
  try:
    # Test case 1: No checkpoints
    result = find_latest_checkpoint_path(temp_dir)
    assert result is None, "Should return None when no checkpoints exist"

    # Test case 2: Single checkpoint
    create_checkpoint_dir(temp_dir, 100)
    result = find_latest_checkpoint_path(temp_dir)
    assert (
        result.resolve() == (Path(temp_dir) / "step=100").resolve()
    ), f"Expected step=100, got {result}"

    # Test case 3: Multiple checkpoints
    create_checkpoint_dir(temp_dir, 50)
    create_checkpoint_dir(temp_dir, 150)
    result = find_latest_checkpoint_path(temp_dir)
    assert (
        result.resolve() == (Path(temp_dir) / "step=150").resolve()
    ), f"Expected step=150, got {result}"

    # Test case 4: Checkpoint with unfinished marker should be ignored
    create_checkpoint_dir(temp_dir, 200, create_unfinished=True)
    result = find_latest_checkpoint_path(temp_dir)
    assert (
        result.resolve() == (Path(temp_dir) / "step=150").resolve()
    ), f"Expected step=150, got {result}"

    # Test case 5: None input
    result = find_latest_checkpoint_path(None)
    assert result is None, "Should return None for None input"

  finally:
    shutil.rmtree(temp_dir)


def test_find_latest_checkpoint_multiple_dirs():
  """Unit test for find_latest_checkpoint_path with multiple directories."""
  temp_dir1 = tempfile.mkdtemp()
  temp_dir2 = tempfile.mkdtemp()
  temp_dir3 = tempfile.mkdtemp()

  try:
    # Test case 1: Empty list
    result = find_latest_checkpoint_path([])
    assert result is None, "Should return None for empty list"

    # Test case 2: List with None values
    result = find_latest_checkpoint_path([None, temp_dir1, None])
    assert result is None, "Should return None when no valid checkpoints exist"

    # Test case 3: Multiple directories with different steps
    create_checkpoint_dir(temp_dir1, 100)
    create_checkpoint_dir(temp_dir2, 200)
    create_checkpoint_dir(temp_dir3, 50)

    result = find_latest_checkpoint_path([temp_dir1, temp_dir2, temp_dir3])
    assert (
        result.resolve() == (Path(temp_dir2) / "step=200").resolve()
    ), f"Expected step=200, got {result}"

    # Test case 4: Some directories have no checkpoints
    empty_dir = tempfile.mkdtemp()
    try:
      result = find_latest_checkpoint_path([temp_dir1, empty_dir, temp_dir2])
      assert (
          result.resolve() == (Path(temp_dir2) / "step=200").resolve()
      ), f"Expected step=200, got {result}"
    finally:
      shutil.rmtree(empty_dir)

  finally:
    shutil.rmtree(temp_dir1)
    shutil.rmtree(temp_dir2)
    shutil.rmtree(temp_dir3)


def run_distributed_checkpoint_test(rank, world_size, temp_dirs, test_case):
  """Run distributed tests for find_latest_checkpoint_path.

  Args:
    rank: Current process rank
    world_size: Total number of processes
    temp_dirs: List of temporary directories for testing
    test_case: Which test case to run
  """
  if test_case == 1:
    # Test synchronization with consistent checkpoints
    temp_dir = temp_dirs[0]
    create_checkpoint_dir(temp_dir, 100)
    create_checkpoint_dir(temp_dir, 200)

    result = find_latest_checkpoint_path(temp_dir, synchronize=True)
    expected = Path(temp_dir) / "step=200"
    assert (
        result.resolve() == expected.resolve()
    ), f"Rank {rank}: Expected {expected}, got {result}"

  elif test_case == 2:
    # Test synchronization when different ranks see different checkpoints
    temp_dir = temp_dirs[0]

    if rank == 0:
      # Rank 0 sees steps 100, 200
      create_checkpoint_dir(temp_dir, 100)
      create_checkpoint_dir(temp_dir, 200)
    else:
      # Other ranks see only step 100
      create_checkpoint_dir(temp_dir, 100)

    # Barrier to ensure all ranks have created their checkpoints
    dist.barrier()

    result = find_latest_checkpoint_path(temp_dir, synchronize=True)
    # With synchronization, all ranks should agree on the maximum step
    expected = Path(temp_dir) / "step=200"
    assert (
        result.resolve() == expected.resolve()
    ), f"Rank {rank}: Expected {expected}, got {result}"

  elif test_case == 3:
    # Test with multiple directories and synchronization
    temp_dir1, temp_dir2 = temp_dirs[0], temp_dirs[1]

    if rank == 0:
      create_checkpoint_dir(temp_dir1, 150)
      create_checkpoint_dir(temp_dir1, 250)

      dist.barrier()

      result = find_latest_checkpoint_path(temp_dir1, synchronize=True)
    else:
      create_checkpoint_dir(temp_dir2, 150)
      create_checkpoint_dir(temp_dir2, 300)

      dist.barrier()

      result = find_latest_checkpoint_path(temp_dir2, synchronize=True)

    expected = Path(temp_dir2) / "step=300"
    assert (
        result.resolve() == expected.resolve()
    ), f"Rank {rank}: Expected {expected}, got {result}"


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is not available"
)
@pytest.mark.parametrize("test_case", [1, 2, 3])
def test_find_latest_checkpoint_distributed(test_case):
  """Distributed tests for find_latest_checkpoint_path synchronization."""
  world_size = 2

  os.environ["MASTER_ADDR"] = "localhost"
  os.environ["MASTER_PORT"] = str(find_available_port())

  # Create temporary directories that will be shared across processes
  temp_dirs = []
  for i in range(2):
    temp_dirs.append(tempfile.mkdtemp(prefix=f"ckpt_test_{test_case}_{i}_"))

  try:
    mp.spawn(
        setup_process,
        args=(
            world_size,
            run_distributed_checkpoint_test,
            temp_dirs,
            test_case,
        ),
        nprocs=world_size,
        join=True,
    )
  finally:
    # Clean up temporary directories
    for temp_dir in temp_dirs:
      if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)


if __name__ == "__main__":
  # Allow running the tests directly
  test_find_latest_checkpoint_input_validation()
  test_find_latest_checkpoint_single_dir()
  test_find_latest_checkpoint_multiple_dirs()
  for test_case in range(1, 4):
    test_find_latest_checkpoint_distributed(test_case)
