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

"""Utilities to support high scale ckpt checkpointing"""

import os
from pathlib import Path
import time
from resiliency.utils import get_resiliency_logger
import yaml

logger = get_resiliency_logger(__name__)

CHECKPOINT_FOLDER_PATH = "/local"
REPLICATOR_CONFIG_FILE_NAME = "replicator.yaml"
REPLICATOR_ERROR_FILE_NAME = "replicator.errors"
REPLICATOR_FAILED_FILE_NAME = "replicator.failed"
REPLICATOR_PRERESTORE_FILE_NAME = "replicator.prerestore"


def _wait_config_file_disappear(config_file_name, timeout_s: int = 300):
  config_file_name = Path(config_file_name)
  for _ in range(timeout_s):
    if not config_file_name.exists():
      logger.info("`replicator.yaml` is accepted.")
      break
    logger.info(f"Waiting for {config_file_name} to disappear")
    time.sleep(1)
  else:
    raise TimeoutError(
        "high scale ckpt existing config does not disappear after"
        f" {timeout_s} seconds"
    )


def init_high_scale_ckpt(
    checkpoint_folder,
    job_name: str,
    gcs_backup_interval_minutes: int = 30,
    peer_ranks: list[int] = None,
    peers_per_node: int = 1,
    blocking: bool = True,
):
  """Generates replicator.yaml file"""
  assert os.getenv("NODE_RANK") is not None, "Please set NODE_RANK envvar"
  assert os.getenv("NNODES") is not None, "Please set NNODES envvar"
  config = {
      "job-name": job_name,
      "framework": "pytorch.distributed",
      "node-rank": int(os.getenv("NODE_RANK")),
      "nodes": int(os.getenv("NNODES")),
      "peers-per-node": peers_per_node,
      "backup-interval-minutes": gcs_backup_interval_minutes,
  }
  if peer_ranks is not None:
    config["peer-ranks"] = peer_ranks
  folder = Path(checkpoint_folder)
  assert folder.exists(), f"_high_scale_ckpt {folder=} doest not exit "
  config_file_name = folder / REPLICATOR_CONFIG_FILE_NAME
  _wait_config_file_disappear(config_file_name)

  with open(config_file_name, "w") as file:
    yaml.dump(config, file, default_flow_style=False, sort_keys=False)
  logger.info(f"Set replicator {config=} as {config_file_name}")


def handle_replicator_fail_situation(directory):
  failed_file_path = Path(directory) / REPLICATOR_FAILED_FILE_NAME
  error_file_path = Path(directory) / REPLICATOR_ERROR_FILE_NAME
  if error_file_path.exists():
    error_message = error_file_path.read_text()
    logger.error(f"Repliactor error: {error_message}")
    error_file_path.unlink()
  if failed_file_path.exists():
    failed_message = failed_file_path.read_text()
    logger.fatal(f"Repliactor failed: {failed_message}")
    failed_file_path.unlink()
    raise ValueError(f'Replictor failed with error {failed_message}')

def process_replicator_prerestore(directory, timeout_s=300):
  """Return step number to be loaded."""

  prerestore_file_path = Path(directory) / REPLICATOR_PRERESTORE_FILE_NAME
  for ts in range(timeout_s):
    if not prerestore_file_path.exists():
      logger.info("waiting for prerestore file.")
      time.sleep(1)
      continue

    with open(prerestore_file_path, "r") as file:
      try:
        prerestore_info = yaml.safe_load(file)
      except yaml.YAMLError as e:
        logger.error(f"Error loading YAML file: {e}")
        return 0
      step = int(prerestore_info.get("restore-step", 0))
      logger.info(f'Replicator found {step=} ckpt available after {ts} s, to be loaded')
      return step
  logger.info(f"Replicator prerestore file not found after {timeout_s} s.")
  raise ValueError(f'Replictor failed to produce prerestore file.')


def block_and_proces_restore_dir(directory, timeout_s=300):
  """Block until a file ending with `.restore` appears, then extract the step number and rename

  the directory using the step number.
  """

  def _extract_step(f):
    # The base file name is formatted as {job_name}-s{step}-n{node_rank}-g{gpu_rank}
    import re

    pattern = r"-s(\d+)-n\d+-w\d+"
    match = re.search(pattern, f.name)
    if not match:
      return 0
    step = int(match.group(1))
    return step

  replicator_config_path = Path(directory) / REPLICATOR_CONFIG_FILE_NAME
  _wait_config_file_disappear(replicator_config_path, timeout_s)
  handle_replicator_fail_situation(directory)

  # TODO: will use this feature when enabling data laoder ckpt.
  process_replicator_prerestore(directory)

  for _ in range(timeout_s):
    restore_files = Path(directory).glob(f"*.restore")
    for f in restore_files:
      step = _extract_step(f)
      if step == 0:
        logger.info("Found a restore directory at step 0, skipping renaming.")
        return None
      target_path = Path(directory) / f"step={step}"
      os.rename(Path(directory) / f, target_path)
      logger.info(
          f"Found a restore directory at step {step} and renamed it from"
          f" {Path(directory) / f} to {target_path}."
      )
      return target_path
    logger.info("waiting for retore file.")
    time.sleep(1)
  raise TimeoutError(
      f"{timeout_s} seconds have passed but no .restore file was found."
  )
