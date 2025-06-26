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
import sys
import torch
import torch.distributed as dist


class SingleLetterFormatter(logging.Formatter):
  """Custom formatter class to convert level names to single letters."""

  def format(self, record):
    record.levelname = record.levelname[0]
    return super().format(record)


class ResiliencyLoggingFormatter(logging.Formatter):

  def format(self, record: logging.LogRecord) -> str:
    """Formats the log record with a custom format.

    Args:
        record (logging.LogRecord): Log record to format.

    Returns:
        str: Formatted log message.
    """
    level_abbr = record.levelname[0]
    module = record.module
    lineno = record.lineno

    timestamp = self.formatTime(record, "%Y-%m-%d %H:%M:%S")
    prefix = f"[Resiliency {level_abbr} {timestamp} {module}:{lineno}]"
    message = super().format(record)
    return f"{prefix} {message}"


class ResiliencyLoggingFilter(logging.Filter):

  def filter(self, record: logging.LogRecord) -> bool:
    """Filter messages based on rank. All logs print on rank 0.

    Debug logs print on all ranks.

    Args:
        record (logging.LogRecord): Log record to filter.

    Returns:
        bool: True if the log should be printed, False otherwise.
    """
    rank = (
        dist.get_rank() if dist.is_initialized() else int(os.getenv("RANK", 0))
    )
    if record.levelno <= logging.DEBUG:
      return True
    return rank == 0


def get_resiliency_logger(
    name: str = "resiliency", level: int = None
) -> logging.Logger:
  """Retrieves resiliency logger.

  Args:
      name (str, optional): Name of logger. Defaults to "resiliency".
      level (int, optional): Logging level. Defaults to logging.INFO.

  Returns:
      logging.Logger: Logger object to use for logging.
  """
  logger = logging.getLogger(name)
  # Unless specified, resiliency logger reverts to basicConfig logging level.
  if level is not None:
    logger.setLevel(level)
  if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    formatter = ResiliencyLoggingFormatter("%(message)s")
    handler.setFormatter(formatter)
    handler.addFilter(ResiliencyLoggingFilter())
    logger.addHandler(handler)

  logger.propagate = False

  return logger


logger = get_resiliency_logger(__name__)


def test_all_reduce(rank, local_rank, world_size):
  """Test all_reduce operation"""
  try:
    tensor = torch.tensor(
        [float(rank)], dtype=torch.float32, device=f"cuda:{local_rank}"
    )
    logger.debug(f"[Rank {rank}] Before all_reduce: {tensor.item()}")
    # Expected sum is sum of all ranks: 0 + 1 + ... + (world_size-1)
    expected = float(world_size * (world_size - 1) / 2)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    logger.debug(
        f"[Rank {rank}] After all_reduce sum: {tensor.item()} expected:"
        f" {expected}"
    )
    assert abs(tensor.item() - expected) < 1e-5
    return True
  except Exception as e:
    logger.debug(f"[Rank {rank}] All_reduce test failed: {str(e)}")
    return False
