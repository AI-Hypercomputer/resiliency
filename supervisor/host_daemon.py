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

from supervisor import host


def get_arg_parser() -> argparse.ArgumentParser:
  """Returns an ArgumentParser with the required flags."""
  parser = argparse.ArgumentParser()
  parser.add_argument(
      "--workers_per_host",
      type=int,
      default=8,
      help="The number of workers expected to run on each host.",
  )
  parser.add_argument(
      "--port", type=int, default=60060, help="The port to listen on."
  )
  parser.add_argument(
      "--address",
      type=str,
      default=None,
      help="The address to listen on.",
  )
  parser.add_argument(
      "--project",
      type=str,
      default="supercomputer-testing",
      help="The GCP project ID.",
  )
  parser.add_argument(
      "--watchdogs",
      action="append",
      type=str,
      choices=["ecc", "xid"],
      help="List of watchdogs to enable. Available options: ecc, xid.",
  )
  parser.add_argument(
      "--watchdog_interval_s",
      type=int,
      default=10,
      help="Interval between fault injections in seconds.",
  )
  parser.add_argument(
      "--watchdog_notification_cooldown_s",
      type=int,
      default=30,
      help="The time between watchdog notifications in seconds.",
  )
  parser.add_argument(
      "--enable_topology_aware_scheduling",
      action="store_true",
      default=False,
      help=(
          "Whether to enable topology aware scheduling. Requires compact"
          " placement."
      ),
  )
  parser.add_argument(
      "--enable_standalone_mode",
      action="store_true",
      default=False,
      help=(
          "Whether to enable standalone mode. When enabled, the host daemon"
          " will check watchdogs for errors without worker heartbeats."
      ),
  )
  return parser


def main():
  parser = get_arg_parser()
  args = parser.parse_args()

  # Create a Host instance
  h = host.Host(
      workers_per_host=args.workers_per_host,
      port=args.port,
      address=args.address,
      project_id=args.project,
      watchdogs=args.watchdogs,
      watchdog_check_interval_s=args.watchdog_interval_s,
      watchdog_notification_cooldown_s=args.watchdog_notification_cooldown_s,
      enable_topology_aware_scheduling=args.enable_topology_aware_scheduling,
      enable_standalone_mode=args.enable_standalone_mode,
  )

  h.start_heartbeat()

  # Wait for workloads to complete
  h.await_completion()

  # Shutdown the Host when done
  h.shutdown()


if __name__ == "__main__":
  main()
