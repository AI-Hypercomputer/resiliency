# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#!/usr/bin/env python3

import os
import sys
import time
import re
import shutil
import argparse
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format=f'[CheckpointCleaner@n{os.getenv("NODE_RANK")}]%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description='Checkpoint cleaner script')
    parser.add_argument('--checkpoint-dir-prefix', required=True,
                       help='Base directory for checkpoints (required)')
    parser.add_argument('--job-identifier', required=True,
                       help='Job identifier for checkpoint directory (required)')
    parser.add_argument('--num-checkpoints-to-keep', type=int, default=2,
                       help='Number of checkpoints to keep (default: 2)')
    parser.add_argument('--sleep-interval', type=int, default=60,
                       help='Sleep interval in seconds (default: 60)')
    return parser.parse_args()

def cleanup_non_matching_jobs(checkpoint_dir_prefix, job_identifier):
    """Clean up non-matching job folders at startup."""
    logger.info("Cleaning up non-matching job folders...")

    try:
        checkpoint_prefix_path = Path(checkpoint_dir_prefix)
        if checkpoint_prefix_path.exists():
            for dir_path in checkpoint_prefix_path.iterdir():
                if dir_path.is_dir() and dir_path.name != job_identifier:
                    logger.info(f"Removing unrelated folder: {dir_path}")
                    try:
                        shutil.rmtree(dir_path)
                    except Exception as e:
                        logger.warning(f"Failed to remove {dir_path}: {e}")
    except Exception as e:
        logger.warning(f"Error during initial cleanup: {e}")

def find_checkpoints(checkpoint_path):
    checkpoint_path = Path(checkpoint_path)
    complete_checkpoints = []
    unfinished_checkpoints = []
    if not checkpoint_path.exists():
        return complete_checkpoints, unfinished_checkpoints

    for item in checkpoint_path.iterdir():
        match = re.search(r'^step=(\d+)$', item.name)
        if not match:
              continue
        step_num = int(match.group(1))
        if (checkpoint_path / f"{item.name}-unfinished").exists():
          logger.debug(f"Found incomplete checkpoint folder: {item.name}")
          unfinished_checkpoints.append((step_num, item.name))
          continue
        complete_checkpoints.append((step_num, item.name))
        logger.debug(f"Found complete checkpoint folder: {item.name}")
    return complete_checkpoints, unfinished_checkpoints

def cleanup_unfinished_checkpoints(checkpoint_path):
    """Clean up unfinished marker files that don't have corresponding complete checkpoint folders."""
    logger.info("Cleaning up unfinished checkpoints...")
    checkpoint_path = Path(checkpoint_path)
    _, unfinished_checkpoints = find_checkpoints(checkpoint_path)

    for step, item_name in unfinished_checkpoints:
        logger.info(f"Found incomplete checkpoint folder: {item_name}")
        (checkpoint_path / f"{item_name}-unfinished").unlink()
        logger.info(f"Removed incomplete checkpoint marker: {item_name}-unfinished")
        shutil.rmtree(checkpoint_path / item_name)
        logger.info(f"Removed incomplete checkpoint folder: {item_name}")

def process_checkpoints(checkpoint_path, num_checkpoints_to_keep):
    """Process and clean up excess checkpoints."""
    # List checkpoints and sort by step number

    complete_checkpoints, unfinished_checkpoints = find_checkpoints(checkpoint_path)

    # Sort by step number
    complete_checkpoints.sort(key=lambda x: x[0])
    unfinished_checkpoints.sort(key=lambda x: x[0])

    complete_names = [name for _, name in complete_checkpoints]
    unfinished_names = [name for _, name in unfinished_checkpoints]

    logger.info(f"Found complete checkpoints: {complete_names}")
    logger.info(f"Found unfinished checkpoints: {unfinished_names}")

    # Remove excess COMPLETE checkpoints if count exceeds num_checkpoints_to_keep
    # We only count complete checkpoints towards the limit
    for i in range(len(complete_checkpoints) - num_checkpoints_to_keep):
        _, oldest_checkpoint = complete_checkpoints[i]
        logger.info(f"Removing old complete checkpoint: {oldest_checkpoint}")

        try:
            # Remove the complete checkpoint folder
            shutil.rmtree(checkpoint_path / oldest_checkpoint)

        except Exception as e:
            logger.warning(f"Failed to remove {oldest_checkpoint}: {e}")
            break


def main():
    args = parse_args()

    checkpoint_dir_prefix = args.checkpoint_dir_prefix
    job_identifier = args.job_identifier
    checkpoint_dir = os.path.join(checkpoint_dir_prefix, job_identifier, 'checkpoint')

    # Set values from arguments
    num_checkpoints_to_keep = args.num_checkpoints_to_keep
    sleep_interval = args.sleep_interval

    # Initial cleanup of non-matching job folders
    cleanup_non_matching_jobs(checkpoint_dir_prefix, job_identifier)
    cleanup_unfinished_checkpoints(checkpoint_dir)

    # Main cleanup loop
    while True:
        # Check if CHECKPOINT_DIR exists and is not empty
        checkpoint_path = Path(checkpoint_dir)
        if not checkpoint_path.exists() or not any(checkpoint_path.iterdir()):
            logger.info("No checkpoints to clean, sleeping...")
            time.sleep(sleep_interval)
            continue

        # Process and clean up checkpoints
        process_checkpoints(checkpoint_path, num_checkpoints_to_keep)

        time.sleep(sleep_interval)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)
