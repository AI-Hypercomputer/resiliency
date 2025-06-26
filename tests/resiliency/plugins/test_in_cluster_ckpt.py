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

import io
import os
from pathlib import Path
import shutil
import socket
import tempfile

from lightning.pytorch.trainer.connectors.checkpoint_connector import (
    _CheckpointConnector,
)
from megatron.core.dist_checkpointing.core import CheckpointingException
from megatron.core.dist_checkpointing.dict_utils import diff
from megatron.core.distributed import DistributedDataParallelConfig
from megatron.core.optimizer import OptimizerConfig
import nemo
from nemo import lightning as nl
from nemo.collections.llm.gpt.data.mock import MockDataModule
from nemo.lightning.pytorch.optim.megatron import MegatronOptimizerModule
from nemo.utils.callbacks.dist_ckpt_io import AsyncFinalizableCheckpointIO
import pytest
from resiliency.callbacks.model_checkpoint import ModelCheckpoint
from resiliency.connectors.checkpoint_connector import CheckpointConnector
from resiliency.model import Llama3Config36M
from resiliency.plugins._ckpt_utils import (
    find_latest_checkpoint_path,
    get_is_checkpoint_file_handler,
)
from resiliency.plugins.in_cluster_local_ckpt import InClusterLocalCheckpointIO
from resiliency.plugins.replication_utils import ReplicatedOptimizerMegatronStrategy
from resiliency.utils import get_resiliency_logger
import torch
import torch.distributed as dist
from torch.distributed.checkpoint.api import CheckpointException
import torch.multiprocessing as mp

logger = get_resiliency_logger(__name__)


def find_available_port() -> int:
  """Find an available port."""
  sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
  sock.bind(("", 0))
  port = sock.getsockname()[1]
  sock.close()
  return port


class NoSyncDirModelCheckpoint(ModelCheckpoint):
  """A ModelCheckpoint class that does not synchronize checkpoint directory across ranks."""

  def setup(self, trainer, *args, **kwargs) -> None:
    pre_sync_dirpath = self.dirpath
    super().setup(trainer, *args, **kwargs)
    self.dirpath = pre_sync_dirpath


class TwoNodeMockCheckpointConnector(CheckpointConnector):
  """A custom checkpoint connector for two-node mock setup."""

  def _restore_modules_and_callbacks(self, checkpoint_path=None) -> None:
    print(f"rank {dist.get_rank()} checkpoint_path:", checkpoint_path)
    checkpoint_dir = None

    if checkpoint_path is not None:
      raise ValueError(
          "Checkpoint path should be None when using the custom checkpoint"
          " connector"
      )

    MAX_TRIALS = 3  # Number of times to try loading a checkpoint per checkpoint directory
    loaded_successfully = False  # Whether a checkpoint was loaded successfully
    start_from_scratch = (
        True  # Whether to start from scratch if no checkpoints are found
    )

    for trial in range(MAX_TRIALS):
      # Find latest checkpoint path on each attempt
      # Use enhanced find_latest_checkpoint_path with list of directories
      checkpoint_dirs_to_search = [
          self.persistent_ckpt_dir,
          self.local_ckpt_dir,
      ]
      checkpoint_path = find_latest_checkpoint_path(
          checkpoint_dir=checkpoint_dirs_to_search,
          synchronize=self.use_ckpt_load_replication,
      )

      # Fix checkpoint path for mock two-node setup
      if checkpoint_path is not None:
        if (
            "node0" in str(checkpoint_path)
            and torch.distributed.get_rank() >= 4
        ):
          checkpoint_path = Path(str(checkpoint_path).replace("node0", "node1"))
        elif (
            "node1" in str(checkpoint_path) and torch.distributed.get_rank() < 4
        ):
          checkpoint_path = Path(str(checkpoint_path).replace("node1", "node0"))

      assert torch.distributed.is_initialized()
      torch.distributed.barrier()
      if checkpoint_path is None:
        break
      start_from_scratch = False
      logger.info(
          f"Trial {trial+1}/{MAX_TRIALS}: Checkpoint loading from"
          f" {checkpoint_path}"
      )

      try:
        _CheckpointConnector._restore_modules_and_callbacks(
            self, checkpoint_path
        )
        if checkpoint_path is not None:
          loaded_successfully = True
          logger.info(f"Successfully loaded checkpoint on trial {trial+1}")
        break  # Exit the loop if loading succeeds
      except (
          CheckpointingException,
          CheckpointException,
          FileNotFoundError,
      ) as e:
        logger.info(
            f"Found invalid checkpoint {checkpoint_path} on trial {trial+1}:"
            f" {str(e)}"
        )
        is_persistent_storage = str(checkpoint_path).startswith(
            str(self.persistent_ckpt_dir)
        )
        if get_is_checkpoint_file_handler(
            is_cluster_local_checkpointing=self.local_ckpt_dir is not None,
            is_persistent_storage=is_persistent_storage,
        ):
          marker_path = (
              ModelCheckpoint.format_checkpoint_unfinished_marker_path(
                  checkpoint_path
              )
          )
          marker_path.parent.mkdir(parents=True, exist_ok=True)
          marker_path.touch()
        logger.info(f"Marked checkpoint {checkpoint_path} as invalid")

        if trial < MAX_TRIALS - 1:
          logger.info(f"Retrying with next available checkpoint...")
        else:
          logger.info(
              f"Exhausted all {MAX_TRIALS} trials for loading checkpoints"
              f" from {checkpoint_dir}"
          )
    if self.persistent_ckpt_dir and get_is_checkpoint_file_handler(
        is_cluster_local_checkpointing=self.local_ckpt_dir is not None,
        is_persistent_storage=True,
    ):
      ModelCheckpoint._remove_unfinished_checkpoints(
          self.persistent_ckpt_dir, True
      )
    if self.local_ckpt_dir and get_is_checkpoint_file_handler(
        is_cluster_local_checkpointing=True, is_persistent_storage=False
    ):
      ModelCheckpoint._remove_unfinished_checkpoints(self.local_ckpt_dir, True)

    if self.terminate_if_load_fail and not (
        loaded_successfully or start_from_scratch
    ):
      raise CheckpointingException(
          f"Failed to load any valid checkpoint after {MAX_TRIALS} attempts"
      )

    if start_from_scratch:
      logger.info("No checkpoints found. Starting training from scratch.")
      super()._restore_modules_and_callbacks(None)
    return None


def setup_process(
    rank,
    world_size,
    test_func,
    base_temp_dir=None,
    test_case=None,
    backend="nccl",
):
  """Initializes the distributed environment and calls the test function."""
  # Simulate 2 nodes with 4 GPUs each
  if rank < 4:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
    os.environ["NODE_RANK"] = "0"
  else:
    os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"
    os.environ["NODE_RANK"] = "1"

  os.environ["RANK"] = str(rank)
  os.environ["LOCAL_RANK"] = str(rank % 4)

  dist.init_process_group(rank=rank, world_size=world_size, backend=backend)

  test_func(rank, world_size, base_temp_dir, test_case)

  dist.destroy_process_group()


def save_load_checkpoint(rank, world_size, base_temp_dir, test_case):
  """Test saving and loading checkpoints with InClusterLocalCheckpointIO and AsyncFinalizableCheckpointIO.

  This function executes a specific test case to verify the behavior of
  replication + in cluster checkpointing logic.
  The test cases will change where files are located upon loading; saving
  behavior will remain constant across all test cases.

  Args:
      rank (int): The rank of the current process.
      world_size (int): The total number of processes.
      base_temp_dir (str): Base temporary directory for checkpoints and logs.
      test_case (int): Which test case to run (1-4)
        1: Every rank has its own checkpoint
        2: Every rank has a different checkpoint
        3: Node0 has checkpoints 0-3, node1 has none
        4: Node1 has checkpoints 0-3, node0 has none
  """
  # TODO: Without this signal handler, this test will fail with a
  # SIGPROF when calling `nemo.collections.llm.train()` the second time.
  # Remove signal handler once the issue is resolved.
  import signal

  signal.signal(signal.SIGPROF, lambda signum, frame: None)

  # Create temporary directory for checkpoints and logs
  if rank < 4:
    node_dir = base_temp_dir / "node0"
  else:
    node_dir = base_temp_dir / "node1"

  node_dir.mkdir(exist_ok=True)

  try:
    ckpt_path = Path(node_dir) / "checkpoints"
    log_path = Path(node_dir) / "logs"
    ckpt_path.mkdir(exist_ok=True)
    log_path.mkdir(exist_ok=True)

    # Create checkpoint IO instance
    checkpoint_io = InClusterLocalCheckpointIO(
        save_ckpt_format="torch_dist",
        load_directly_on_device=True,
        async_save=True,
        torch_dist_multiproc=2,  # Use 2 threads per rank
        assume_constant_structure=True,
        parallel_save=False,
        parallel_save_within_dp=False,
        parallel_load=False,
        use_ckpt_load_replication=True,
    )

    # Wrap checkpoint_io with persistent checkpoint process
    checkpoint_io = AsyncFinalizableCheckpointIO(checkpoint_io)

    # Define model, optimizer, and data
    model_config = Llama3Config36M()
    model = nemo.collections.llm.LlamaModel(model_config)

    opt_config = OptimizerConfig(
        optimizer="adam",
        lr=1e-4,
        weight_decay=0.1,
        adam_beta1=0.9,
        adam_beta2=0.95,
        adam_eps=1e-8,
        clip_grad=1.0,
        log_num_zeros_in_grad=False,
        timers=None,
        bf16=True,
        use_distributed_optimizer=True,
    )
    optim = MegatronOptimizerModule(config=opt_config)

    data = MockDataModule(
        seq_length=model_config.seq_length,
        global_batch_size=2,
        num_train_samples=100,
        pin_memory=False,
        micro_batch_size=1,
    )

    # Define megatron strategy
    strategy = ReplicatedOptimizerMegatronStrategy(
        tensor_model_parallel_size=4,
        context_parallel_size=1,
        pipeline_dtype=torch.bfloat16,
        sequence_parallel=False,
        ckpt_async_save=True,
        ddp=DistributedDataParallelConfig(
            num_distributed_optimizer_instances=2
        ),
    )

    # Define logger and callbacks
    nemo_logger = nl.NeMoLogger(
        log_dir=log_path,
        use_datetime_version=False,
        update_logger_directory=True,
        wandb=None,
    )

    # Since we are mocking two devices, we need to override dirpath sync
    callbacks = [
        NoSyncDirModelCheckpoint(
            dirpath=ckpt_path,
            save_last=False,
            monitor="step",
            save_top_k=-1,
            mode="max",
            save_weights_only=False,
            every_n_train_steps=1,
            save_on_train_epoch_end=True,
            save_optim_on_train_end=True,
            always_save_context=False,
            filename="{step}",
            enable_version_counter=False,
            use_in_cluster_local_ckpts=True,
            preprocess_files=True,
        )
    ]

    # Create trainer
    trainer = nl.Trainer(
        accelerator="gpu",
        devices=4,
        num_nodes=world_size // torch.cuda.device_count(),
        max_steps=2,
        plugins=[
            checkpoint_io,
            nl.MegatronMixedPrecision(precision="bf16-mixed"),
        ],
        strategy=strategy,
        log_every_n_steps=None,
        val_check_interval=None,
        limit_val_batches=None,
        callbacks=callbacks,
    )

    # Define custom checkpoint connector to support in-cluster checkpointing
    trainer._checkpoint_connector = TwoNodeMockCheckpointConnector(
        trainer=trainer,
        local_ckpt_dir=ckpt_path,
        use_ckpt_load_replication=True,
    )

    # Train for 2 steps to generate 2 checkpoints
    train_result = nemo.collections.llm.train(
        model=model,
        data=data,
        trainer=trainer,
        log=nemo_logger,
        optim=optim,
        tokenizer="data",
    )

    # Verify checkpoint files exist
    for i in range(2):
      assert (
          ckpt_path / f"step={i}"
      ).exists(), f"Checkpoint directory step={i+1} doesn't exist"
      # Check for checkpoint files
      checkpoint_files = list((ckpt_path / f"step={i}").glob("*"))

    # Create a new model and load the checkpoint
    new_model = nemo.collections.llm.LlamaModel(model_config)
    new_optim = MegatronOptimizerModule(config=opt_config)

    # Shuffle around files as needed for test cases
    if test_case == 1:
      # For test case 1, every rank has its own checkpoint
      pass  # Checkpoint files do not need to be moved
    elif test_case == 2:
      # For test case 2, shuffle around checkpoint files between nodes
      dist.barrier()  # Ensure all ranks have finished saving
      if rank == 0:  # Only one process should handle the swap

        # Get node paths
        node0_dir = base_temp_dir / "node0"
        node1_dir = base_temp_dir / "node1"

        # Create a temporary directory to hold the files during the swap
        temp_dir = base_temp_dir / "temp_swap"

        # Swap directories
        os.rename(str(node0_dir), str(temp_dir))
        os.rename(str(node1_dir), str(node0_dir))
        os.rename(str(temp_dir), str(node1_dir))

      dist.barrier()  # Ensure all ranks see the updated files
    elif test_case == 3:
      # For test case 3, node0 has checkpoints 0-3, node1 has none
      dist.barrier()  # Ensure all ranks have finished saving
      if rank == 0:
        shutil.rmtree(base_temp_dir / "node1")
      dist.barrier()  # Ensure all ranks see the updated files
    elif test_case == 4:
      # For test case 4, node1 has checkpoints 0-3, node0 has none
      dist.barrier()  # Ensure all ranks have finished saving
      if rank == 0:
        shutil.rmtree(base_temp_dir / "node1")
        os.rename(str(base_temp_dir / "node0"), str(base_temp_dir / "node1"))
      dist.barrier()  # Ensure all ranks see the updated files

    # Call train again to load the checkpoint into the new model and optimizer objects
    train_result = nemo.collections.llm.train(
        model=new_model,
        data=data,
        trainer=trainer,
        log=nemo_logger,
        optim=new_optim,
        tokenizer="data",
        resume=None,
    )

    # Verify that the new model has the same parameters as the original
    orig_model_sd = model.state_dict()
    new_model_sd = new_model.state_dict()
    model_diffs = diff(orig_model_sd, new_model_sd)

    # Diffs have structure of (key only in arg1, key only in arg2, key in both arg1 and arg2 with different values)
    assert (
        len(model_diffs[0]) == len(model_diffs[1]) == 0
    ), "Model state dicts differ"

    for k, _, _ in model_diffs[2]:
      k = k[0]
      v1 = orig_model_sd[k]
      v2 = new_model_sd[k]

      assert isinstance(v1, io.BytesIO) and isinstance(
          v2, io.BytesIO
      ), f"Model state dicts differ at key: {k}"
      assert (
          v1.getbuffer() == v2.getbuffer()
      ), f"Model state dicts differ at key: {k}"

    # Verify that the new optimizer has the same parameters as the original
    orig_optim_sd = model.state_dict()
    new_optim_sd = new_model.state_dict()
    optimizer_diffs = diff(orig_optim_sd, new_optim_sd)

    assert (
        len(optimizer_diffs[0]) == len(optimizer_diffs[1]) == 0
    ), "Optimizer state dicts differ"

    for k, _, _ in optimizer_diffs[2]:
      k = k[0]
      v1 = orig_optim_sd[k]
      v2 = new_optim_sd[k]

      assert isinstance(v1, io.BytesIO) and isinstance(
          v2, io.BytesIO
      ), f"Optimizer state dicts differ at key: {k}"
      assert (
          v1.getbuffer() == v2.getbuffer()
      ), f"Optimizer state dicts differ at key: {k}"

  finally:
    # Clean up temporary directory
    dist.barrier()
    if rank == 0:
      shutil.rmtree(base_temp_dir)


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is not available"
)
@pytest.mark.parametrize("test_case", [1, 2, 3, 4])
def test_save_load_checkpoint(test_case):
  world_size = 8

  assert world_size == torch.cuda.device_count(), (
      "This test requires 8 GPUs, "
      f"but only {torch.cuda.device_count()} are available."
  )

  os.environ["MASTER_ADDR"] = "localhost"
  os.environ["MASTER_PORT"] = str(find_available_port())

  base_temp_dir = Path(tempfile.mkdtemp(prefix=""))

  # Run the specific test case
  mp.spawn(
      setup_process,
      args=(world_size, save_load_checkpoint, base_temp_dir, test_case),
      nprocs=world_size,
      join=True,
  )


if __name__ == "__main__":
  for test_case in range(1, 5):
    test_save_load_checkpoint(test_case)
