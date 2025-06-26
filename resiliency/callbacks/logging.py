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

from datetime import datetime
import os
import time
from lightning.pytorch.trainer.connectors.checkpoint_connector import (
    _CheckpointConnector,
)
import nemo
import psutil
import torch
import torch.distributed as dist

if nemo.__version__.startswith("2.1.0"):
  import lightning.pytorch as pl
else:
  import pytorch_lightning as pl

from resiliency.utils import get_resiliency_logger
from tensorboardX import SummaryWriter

logger = get_resiliency_logger(__name__)


class MemoryLoggingCallback(pl.Callback):

  def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
    logger.info("Memory usage after model/optimizer setup:")
    self._log_cpu_mem()
    self._log_model_and_optimizer_cuda_memory(pl_module, trainer)

  def setup(self, trainer, pl_module, stage=None):
    logger.info("Memory usage before model/optimizer setup:")
    self._log_cpu_mem()
    self._log_model_and_optimizer_cuda_memory(pl_module, trainer)

  def _log_cpu_mem(self):
    mem = psutil.virtual_memory()
    logger.info(
        f"CPU Memory Usage: {mem.used / 1024**3:.2f} GB used /"
        f" {mem.total / 1024**3:.2f} GB total"
    )

  def _log_model_and_optimizer_cuda_memory(self, pl_module, trainer):
    torch.cuda.synchronize()

    def tensor_size_gb(tensor):
      return (
          tensor.element_size() * tensor.nelement() / 1024**3
          if tensor.is_cuda
          else 0
      )

    # Estimate model CUDA memory usage
    # This only calculates for first virtual pipeline stage
    # if virtual pipeline parallelism is enabled
    model_cuda_mem = 0.0
    for name, param in pl_module.named_parameters(remove_duplicate=False):
      if param.is_cuda:
        size_gb = tensor_size_gb(param)
        model_cuda_mem += size_gb
    estiamted_model_cuda_mem = (
        model_cuda_mem * pl_module.config.virtual_pipeline_model_parallel_size
    )

    # Estimate optimizer CUDA memory usage
    optim_cuda_mem = 0.0
    for opt_idx, optimizer in enumerate(trainer.optimizers):
      for group in optimizer.param_groups:
        for p in group["params"]:
          state = optimizer.state[p]
          for k, v in state.items():
            if isinstance(v, torch.Tensor) and v.is_cuda:
              size_gb = tensor_size_gb(v)
              optim_cuda_mem += size_gb

    logger.info(
        f"Estimated model CUDA memory usage: {estiamted_model_cuda_mem:.2f} GB"
    )
    logger.info(
        f"Estimated optimizer CUDA memory usage: {optim_cuda_mem:.2f} GB"
    )

    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    logger.info(
        f"Actual CUDA memory usage: {allocated:.2f} GB allocated /"
        f" {reserved:.2f} GB reserved"
    )


class TensorboardCallback(pl.Callback):

  def __init__(self, tb_dir):
    self.tb_dir = tb_dir
    self._tb_writer = None

  def setup_tensorboard(self):
    if self.tb_dir and dist.get_rank() == 0:
      self._tb_writer = SummaryWriter(logdir=self.tb_dir)

  def teardown_tensorboard(self):
    if self._tb_writer:
      self._tb_writer.close()

  def on_fit_start(self, trainer, pl_module):
    self.setup_tensorboard()

  def on_fit_end(self, trainer, pl_module):
    self.teardown_tensorboard()

  def optionally_write(self, tag, value, step):
    if self._tb_writer:
      self._tb_writer.add_scalar(tag, value, step)


class StepLoggingCallback(TensorboardCallback):

  def __init__(self, tb_dir=None, timer_kwargs={}):
    if tb_dir:
      tb_dir = os.path.join(
          tb_dir, f"/run-{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
      )
    super().__init__(tb_dir)
    self._sync_cuda = timer_kwargs.get("sync_cuda", False)
    self.last_step_end = None

  def _sync(self):
    if self._sync_cuda and torch.cuda.is_initialized():
      torch.cuda.synchronize()

  def on_fit_start(self, trainer, pl_module):
    super().on_fit_start(trainer, pl_module)
    current_time = time.time()
    logger.info("Training starts.")
    self.last_step_end = current_time

  def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
    self._sync()
    current_time = time.time()
    if self.last_step_end:
      between_steps = current_time - self.last_step_end
      if between_steps > 0.01:
        logger.info(
            "Time between last step end and this step start:"
            f" {between_steps:.3f}s"
        )
        self.optionally_write(
            "train/between_steps", between_steps, trainer.global_step
        )

    self.step_start = current_time

  def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
    self._sync()
    step_end = time.time()
    step_time = step_end - self.step_start
    logger.info(f"Step {trainer.global_step} time: {step_time:.3f}s")
    self.optionally_write("train/step_time", step_time, trainer.global_step)
    self.optionally_write(
        "train/num_gpus", trainer.world_size, trainer.global_step
    )
    self.last_step_end = step_end

  def on_fit_end(self, trainer, pl_module):
    self._sync()
    current_time = time.time()
    if self.last_step_end:
      between_steps = current_time - self.last_step_end
      logger.info(
          f"Time between steps {trainer.global_step} end and train"
          f" end: {between_steps:.3f}s"
      )

      super().on_fit_end(trainer, pl_module)


class TPSLoggingCallback(TensorboardCallback):

  def __init__(self, gbs, seq_length, tb_dir=None):
    if tb_dir:
      tb_dir = os.path.join(
          tb_dir, f"/tps-{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
      )
    super().__init__(tb_dir)
    self.prev_step_start = None
    self.step_end = None
    self.gbs = gbs
    self.seq_length = seq_length

  def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
    if self.prev_step_start:
      step_time = time.time() - self.prev_step_start
      tps = self.gbs * self.seq_length / step_time
      logger.info(f"TPS for step {trainer.global_step}: {tps:.3f}")
      self.optionally_write("train/tps", tps, trainer.global_step)

    self.prev_step_start = time.time()
