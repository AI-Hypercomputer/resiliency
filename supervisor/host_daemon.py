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

from supervisor import host


def main():
  # Replace with your actual GCP project ID
  project_id = str(os.environ.get("PROJECT", "supercomputer-testing"))

  # Port for the Host to listen on
  port = int(os.environ.get("HOST_DAEMON_PORT", 60060))

  # Create a Host instance
  h = host.Host(project_id, port, watchdogs=["ecc", "xid"])

  h.start_heartbeat()

  # Wait for workloads to complete
  h.await_completion()

  # Shutdown the Host when done
  h.shutdown()


if __name__ == "__main__":
  main()
