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
import time
from typing import Any, Dict, Optional, Union
import nemo
import torch

if nemo.__version__.startswith("2.1.0"):
  import lightning.pytorch as pl
else:
  import pytorch_lightning as pl

from lightning.pytorch.trainer.connectors.checkpoint_connector import (
    _CheckpointConnector,
)
import resiliency.high_scale_ckpt_utils as high_scale_ckpt_utils
from resiliency.plugins._ckpt_utils import get_is_checkpoint_file_handler, find_latest_checkpoint_path
from megatron.core.dist_checkpointing.core import CheckpointingException
from pathlib import Path
from resiliency.callbacks import model_checkpoint
from torch.distributed.checkpoint.api import CheckpointException
from resiliency.goodput_measure import logging as goodput_logging
from resiliency.goodput_measure import constant as goodput_event

from resiliency.utils import get_resiliency_logger

logger = get_resiliency_logger(__name__)


def _get_iteration_from_checkpoint(checkpoint: Dict[str, Any]) -> Optional[int]:
  return (
      checkpoint.get("loops", {})
      .get("fit_loop", {})
      .get("epoch_loop.batch_progress", {})
      .get("total", {})
      .get("processed", 0)
  )


class CheckpointConnector(_CheckpointConnector):

  def __init__(
      self,
      trainer: "pl.Trainer",
      enable_high_scale_ckpt=False,
      use_ckpt_load_replication=False,
      local_ckpt_dir=None,
      persistent_ckpt_dir=None,
      terminate_if_load_fail=True,
  ) -> None:
    super().__init__(trainer)
    self.enable_high_scale_ckpt = enable_high_scale_ckpt
    self.use_ckpt_load_replication = use_ckpt_load_replication
    if local_ckpt_dir is None and persistent_ckpt_dir is None:
      raise ValueError(
          "Checkpoint directory must be specified. "
          "Please set either local_ckpt_dir or persistent_ckpt_dir."
      )
    self.local_ckpt_dir = local_ckpt_dir
    self.persistent_ckpt_dir = persistent_ckpt_dir
    self.terminate_if_load_fail = terminate_if_load_fail

  def resume_start(self, checkpoint_path=None) -> None:
    logger.info(
        "Finish runtime and model init, starting checkpoint loading from"
        f" {checkpoint_path}"
    )
    super().resume_start(checkpoint_path)

  def resume_end(self) -> None:
    step = _get_iteration_from_checkpoint(self._loaded_checkpoint)
    self.trainer.fit_loop.epoch_loop._batches_that_stepped
    super().resume_end()
    logger.info(f"Finish checkpoint loading, resumed from {step=}")
    if os.getenv("RANK") == "0":
      goodput_logging.log_event(goodput_event.CHECKPOINT_LOADED, step=step)

  def _restore_modules_and_callbacks(self, checkpoint_path=None) -> None:
    checkpoint_dir = None

    if checkpoint_path is not None:
      raise ValueError(
          "Checkpoint path should be None when using the custom checkpoint"
          " connector"
      )

    if self.enable_high_scale_ckpt:
      checkpoint_dir = high_scale_ckpt_utils.CHECKPOINT_FOLDER_PATH
      if get_is_checkpoint_file_handler(is_cluster_local_checkpointing=True):
        high_scale_ckpt_utils.block_and_proces_restore_dir(
            checkpoint_dir, timeout_s=300
        )
      assert torch.distributed.is_initialized()
      torch.distributed.barrier()

    MAX_TRIALS = 3  # Number of times to try loading a checkpoint per checkpoint directory
    loaded_successfully = False  # Whether a checkpoint was loaded successfully
    start_from_scratch = (
        True  # Whether to start from scratch if no checkpoints are found
    )

    for trial in range(MAX_TRIALS):
      # Find latest checkpoint path on each attempt
      if self.enable_high_scale_ckpt:
        checkpoint_path = find_latest_checkpoint_path(
            checkpoint_dir=checkpoint_dir,
            synchronize=self.use_ckpt_load_replication,
        )
      else:
        # Last trial will only search persistent storage
        if trial + 1 == MAX_TRIALS:
          logger.info("Last try, searching only persistent storage")
          checkpoint_dirs_to_search = self.persistent_ckpt_dir
        else:
          checkpoint_dirs_to_search = [
              self.persistent_ckpt_dir,
              self.local_ckpt_dir,
          ]

        # Find latest checkpoint path in the specified directories
        checkpoint_path = find_latest_checkpoint_path(
            checkpoint_dir=checkpoint_dirs_to_search,
            synchronize=self.use_ckpt_load_replication,
        )

      is_cluster_local_checkpointing = str(checkpoint_path).startswith(
          str(self.local_ckpt_dir)
      )

      assert torch.distributed.is_initialized()
      torch.distributed.barrier()
      if checkpoint_path is None:
        logger.info("No valid checkpoint found, start from scratch..")
        start_from_scratch = True
        break
      start_from_scratch = False
      logger.info(
          f"Trial {trial+1}/{MAX_TRIALS}: Checkpoint loading from"
          f" {checkpoint_path}"
      )

      try:

        super()._restore_modules_and_callbacks(checkpoint_path)
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
        if get_is_checkpoint_file_handler(
            is_cluster_local_checkpointing=is_cluster_local_checkpointing,
        ):
          marker_path = model_checkpoint.ModelCheckpoint.format_checkpoint_unfinished_marker_path(
              checkpoint_path
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
        is_cluster_local_checkpointing=is_cluster_local_checkpointing,
    ):
      model_checkpoint.ModelCheckpoint._remove_unfinished_checkpoints(
          self.persistent_ckpt_dir, True
      )
    if self.local_ckpt_dir and get_is_checkpoint_file_handler(
        is_cluster_local_checkpointing=is_cluster_local_checkpointing,
    ):
      model_checkpoint.ModelCheckpoint._remove_unfinished_checkpoints(
          self.local_ckpt_dir, True
      )

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
