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

import logging
import os
import signal
import sys
import nemo
import torch.distributed as dist

if nemo.__version__.startswith("2.1.0"):
  import lightning.pytorch as pl
else:
  import pytorch_lightning as pl
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from nemo.utils.callbacks.dist_ckpt_io import AsyncFinalizableCheckpointIO
from resiliency.utils import get_resiliency_logger

logger = get_resiliency_logger(__name__)


class TCPStoreSignalHandler:

  def __init__(self, store):
    self.rank = dist.get_rank()
    self.received_signal = False
    self.store = store
    self.shutdown_key = "shutdown_signal"
    self.ready_key_prefix = "ready_to_save_"

    if self.rank == 0:
      self.store.set(self.shutdown_key, "0")

    signal.signal(signal.SIGTERM, self._handle_signal)

  def _handle_signal(self, signum, frame):
    self.received_signal = True
    logger.info(f"Process {self.rank} received signal {signum}")
    try:
      self.store.set(self.shutdown_key, "1")
    except Exception as e:
      logger.error(f"Failed to set shutdown signal: {e}")

  def mark_ready_to_save(self):
    """Mark this rank as ready to save checkpoint"""
    self.store.set(f"{self.ready_key_prefix}{self.rank}", "1")

  def should_stop(self):
    """Check if should stop and all ranks are ready"""
    try:
      # First check shutdown signal
      if self.store.get(self.shutdown_key) == b"1":
        self.received_signal = True
        logger.info(f"Process {self.rank} detected shutdown signal")
        return True
        # Mark this rank as ready
        self.mark_ready_to_save()

        # Wait for all ranks to be ready
        world_size = dist.get_world_size()
        ready_count = 0
        for r in range(world_size):
          try:
            if self.store.get(f"{self.ready_key_prefix}{r}") == b"1":
              ready_count += 1
          except Exception:
            pass

        if ready_count == world_size:
          return True

      return False

    except Exception as e:
      logger.error(f"Failed to check signals: {e}")
      return False


class AutoCheckpointCallback(pl.Callback):

  def __init__(self, checkpoint_dir):
    super().__init__()
    self.checkpoint_dir = checkpoint_dir
    self.signal_handler = None
    os.makedirs(checkpoint_dir, exist_ok=True)

  def on_train_start(self, trainer, pl_module):
    store = (
        dist.distributed_c10d._get_default_store()
    )  # _get_default_group().store
    self.signal_handler = TCPStoreSignalHandler(store=store)

  def _finalize_if_async_save(self, trainer):
    checkpoint_io = trainer.strategy.checkpoint_io
    if not isinstance(checkpoint_io, AsyncFinalizableCheckpointIO):
      return
    checkpoint_io.maybe_finalize_save_checkpoint(blocking=True)

  def get_checkpoint_callback(self, trainer):
    callbacks = [c for c in trainer.callbacks if isinstance(c, ModelCheckpoint)]
    return callbacks[0] if len(callbacks) > 0 else None

  def on_train_batch_end(
      self, trainer, pl_module, outputs, batch, batch_idx, unused=0
  ):
    if self.signal_handler and self.signal_handler.should_stop():
      logger.info(f"Saving emergency ckpt.")
      checkpoint_callback = self.get_checkpoint_callback(trainer)
      monitor_candidates = checkpoint_callback._monitor_candidates(trainer)
      checkpoint_callback._save_topk_checkpoint(trainer, monitor_candidates)
      self._finalize_if_async_save(trainer)
      logger.info("Emergency checkpoint saved.")
      trainer.strategy.barrier()  # Ensure all processes are synced
      logger.info("Exiting Training.")
      sys.exit(0)
