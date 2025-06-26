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

from supervisor import sensor
import supervisor_core


def main():
  port = int(os.environ.get("PORT", 90091))  # Default to 90091 if not set
  num_grpc_threads = 32

  # Create a SupervisorConfig instance using environment variables
  config = supervisor_core.SupervisorConfig.from_environment()
  sensor_instance = sensor.Sensor(port, num_grpc_threads, config)

  # Wait for workloads to complete
  sensor_instance.wait_for_completion()

  # Shutdown the Sensor when done
  sensor_instance.shutdown()


if __name__ == "__main__":
  main()
