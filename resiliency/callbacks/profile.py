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

import nemo

if nemo.__version__.startswith("2.1.0"):
  import lightning.pytorch as pl
else:
  import pytorch_lightning as pl
from resiliency.utils import get_resiliency_logger
import torch

logger = get_resiliency_logger(__name__)

MAX_TRACE_ENTRIES = 500_000_000


class ProfileCheckpointCallback(pl.Callback):

  def __init__(self, trace_dir, profile_checkpoint_interval):
    self.log_dir = trace_dir
    self.profile_checkpoint_interval = profile_checkpoint_interval
    self.rank = torch.distributed.get_rank()
    self.profile_is_on = False

  def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
    # when we call batch_end, gobal_step already increased by 1
    if self.rank == 0 and trainer.global_step > 1:
      if trainer.global_step % self.profile_checkpoint_interval == 0:
        from viztracer import VizTracer

        self.tracer = VizTracer(
            tracer_entries=MAX_TRACE_ENTRIES, max_stack_depth=50, log_torch=True
        )
        self.tracer.start()
        self.profile_is_on = True
        logger.info(f"Started Checkpoint profile")

  def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
    if not hasattr(self, "tracer") or self.tracer is None:
      return
    if self.rank == 0 and trainer.global_step > 1:
      if (
          self.profile_is_on
          and trainer.global_step % self.profile_checkpoint_interval == 0
      ):
        self.tracer.stop()
        self.tracer.save(
            output_file=f"{self.trace_dir}/ckpt_rank{self.rank}_step{trainer.global_step}_trace.json"
        )
        self.profile_is_on = False
        logger.info(f"Finished Checkpoint profile")

  def on_fit_end(self, trainer, pl_module):
    if self.profile_is_on:
      self.tracer.stop()
      self.tracer.save(
          output_file=f"{self.trace_dir}/ckpt_rank{self.rank}_step{trainer.global_step}_trace.json"
      )
      self.profile_is_on = False
      logger.info(f"Finished Checkpoint profile")
