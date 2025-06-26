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

from supervisor import actuator
import supervisor_core


def main():
  port = int(os.environ.get("PORT", 90090))
  num_grpc_threads = 32

  # Create a SupervisorConfig instance using environment variables
  config = supervisor_core.SupervisorConfig.from_environment()

  actuator_instance = actuator.Actuator(port, num_grpc_threads, config)

  # Wait for workloads to complete
  actuator_instance.wait_for_completion()

  # Shutdown the Actuator when done
  actuator_instance.shutdown()


if __name__ == "__main__":
  main()
