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

import argparse
import logging
import random
import time

from simulator import gpu_failure_simulator, gpu_slowdown_simulator, xid_simulator
from torch.distributed.argparse_util import env


def get_arg_parser() -> argparse.ArgumentParser:
  """Returns an ArgumentParser with the required flags."""
  parser = argparse.ArgumentParser()
  parser.add_argument(
      "--node_rank",
      action=env,
      type=int,
      default=0,
      help="Node rank.",
  )
  parser.add_argument(
      "--nnodes",
      type=int,
      default=1,
      help="Total number of nodes.",
  )
  parser.add_argument(
      "--gpus_per_node",
      type=int,
      default=8,
      help="Number of GPUs per node.",
  )
  parser.add_argument(
      "--fault_interval",
      type=int,
      default=10,
      help="Interval between fault injections.",
  )
  parser.add_argument(
      "--enable_gpu_failure",
      action="store_true",
      default=False,
      help="Enable GPU failure injection.",
  )
  parser.add_argument(
      "--enable_gpu_slowdown",
      action="store_true",
      default=False,
      help="Enable GPU slowdown injection.",
  )
  parser.add_argument(
      "--gpu_slowdown_duration_s",
      type=int,
      default=30,
      help="Duration of GPU slowdown injection in seconds.",
  )
  parser.add_argument(
      "--enable_xid_failure",
      action="store_true",
      default=False,
      help="Enable XID failure injection.",
  )
  parser.add_argument(
      "--seed",
      action=env,
      type=int,
      default=42,
      help="Seed for random number generator.",
  )
  return parser


def setup_logger() -> logging.Logger:
  """Sets up the logger with a specific prefix format.

  Returns:
      logging.Logger: The logger object.
  """
  logger = logging.getLogger(__name__)
  logger.setLevel(logging.INFO)

  # Check if a handler already exists
  if not logger.handlers:
    formatter = logging.Formatter(
        "%(levelname)s %(asctime)s  %(process)d %(filename)s:%(lineno)d]"
        " %(message)s"
    )
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    logger.addHandler(handler)

  logger.propagate = False
  return logger


def main():
  parser = get_arg_parser()
  logger = setup_logger()
  args = parser.parse_args()
  random.seed(args.seed)

  logger.info(
      "Starting Google Cloud Resiliency fault simulator with seed %d.",
      args.seed,
  )

  gpu_world_size = args.nnodes * args.gpus_per_node
  sim_list = []
  if args.enable_gpu_failure:
    logger.info("Enabling GPU failure injection.")
    sim_list.append(
        gpu_failure_simulator.GPUFailureSimulator(
            args.node_rank,
            gpu_world_size,
            gpus_per_node=args.gpus_per_node,
            seed=args.seed,
        )
    )
  if args.enable_gpu_slowdown:
    logger.info("Enabling GPU slowdown injection.")
    sim_list.append(
        gpu_slowdown_simulator.GPUSlowdownSimulator(
            args.node_rank,
            gpu_world_size,
            gpus_per_node=args.gpus_per_node,
            seed=args.seed,
        )
    )
  if args.enable_xid_failure:
    logger.info("Enabling XID failure injection.")
    sim_list.append(
        xid_simulator.XIDSimulator(
            args.node_rank,
            gpu_world_size,
            gpus_per_node=args.gpus_per_node,
            seed=args.seed,
        )
    )

  is_completed = False
  while not is_completed:
    logger.info("Sleeping for %d seconds.", args.fault_interval)
    time.sleep(args.fault_interval)

    sim = random.choice(sim_list)
    global_gpu_rank = random.randint(0, gpu_world_size - 1)

    logger.info(
        "Injecting failure using simulator %s on GPU %d.",
        sim.__class__.__name__,
        global_gpu_rank,
    )

    if isinstance(sim, xid_simulator.XIDSimulator):
      sim.induce_event(global_ranks=[global_gpu_rank], should_crash=True)
    elif isinstance(sim, gpu_slowdown_simulator.GPUSlowdownSimulator):
      sim.induce_event(
          global_ranks=[global_gpu_rank],
          duration=args.gpu_slowdown_duration_s,
          gpu_power_limit=200,
      )
    else:
      sim.induce_event(global_ranks=[global_gpu_rank])

    is_completed = sim.is_completed()


if __name__ == "__main__":
  main()
