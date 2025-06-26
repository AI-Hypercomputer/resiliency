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

from dataclasses import dataclass
import io
import os
from pathlib import Path
import shutil
import tempfile
from typing import Optional
import pytest

from megatron.core.dist_checkpointing.dict_utils import diff
from megatron.core.distributed import DistributedDataParallelConfig
from megatron.core.optimizer import OptimizerConfig
import nemo
from nemo import lightning as nl
from nemo.collections.llm.gpt.data.mock import MockDataModule
from nemo.lightning.pytorch.optim.megatron import MegatronOptimizerModule
from resiliency.callbacks import model_checkpoint
from resiliency.model import Llama3Config36M
from resiliency.plugins._ckpt_utils import find_latest_checkpoint_path
from resiliency.plugins.min_ckpt_overhead import MinCkptOverheadCheckpointIO
from resiliency.plugins.persistent_ckpt_proc import PersistentCheckpointProcessIO
import torch



@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is not available"
)
def test_save_load_checkpoint():
  """Test saving and loading checkpoints with MinCkptOverheadCheckpointIO and PersistentCheckpointProcessIO."""
  # Set RANK environment variable
  os.environ["RANK"] = "0"

  # Create temporary directory for checkpoints and logs
  temp_dir = tempfile.mkdtemp()
  try:
    ckpt_path = Path(temp_dir) / "checkpoints"
    log_path = Path(temp_dir) / "logs"
    ckpt_path.mkdir(exist_ok=True)
    log_path.mkdir(exist_ok=True)

    # Create checkpoint IO instance
    checkpoint_io = MinCkptOverheadCheckpointIO(
        save_ckpt_format="torch_dist",
        load_directly_on_device=True,
        async_save=True,
        torch_dist_multiproc=2,  # Use 2 threads per rank
        assume_constant_structure=True,
        parallel_save=True,
        parallel_save_within_dp=False,
        parallel_load=True,
    )

    # Wrap checkpoint_io with persistent checkpoint process
    checkpoint_io = PersistentCheckpointProcessIO(
        checkpoint_io,
    )

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
    strategy = nl.MegatronStrategy(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        pipeline_dtype=torch.bfloat16,
        sequence_parallel=False,
        ckpt_async_save=True,
        ddp=DistributedDataParallelConfig(),
    )

    # Define logger and callbacks
    nemo_logger = nl.NeMoLogger(
        log_dir=log_path,
        use_datetime_version=False,
        update_logger_directory=True,
        wandb=None,
    )

    callbacks = [
        model_checkpoint.ModelCheckpoint(
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
        )
    ]

    # Create trainer
    trainer = nl.Trainer(
        accelerator="gpu",
        devices=1,
        max_steps=2,
        plugins=[checkpoint_io],
        strategy=strategy,
        log_every_n_steps=None,
        val_check_interval=None,
        limit_val_batches=None,
        callbacks=callbacks,
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
    if os.path.exists(temp_dir):
      shutil.rmtree(temp_dir)


if __name__ == "__main__":
  test_save_load_checkpoint()
