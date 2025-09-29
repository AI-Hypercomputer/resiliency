# Quickstart: Integrating the Supervisor Client with PyTorch
This guide provides a step-by-step tutorial for integrating the `supervisor` client into a PyTorch workload using `torchrun`. By following these instructions, you can enable communication between your training workers and the `supervisor` service to report their state without modifying your core training code. This integration allows `supervisor` to monitor your workload and perform actions like live-swapping faulty hardware.

The integration is achieved by patching PyTorch's `LocalElasticAgent` to include the supervisor client, which will then report the worker's status (`HEALTHY`, `FAILED`, `SUCCEEDED`).

## Prerequisites
Before you begin, ensure you have the following set up:

 1. A running Google Kubernetes Engine (GKE) cluster.

 2. The supervisor service components deployed on your GKE cluster. If you haven't done this, please follow the official deployment guide.

## Step 1: Create the Patch File
First, you need to create a diff file to patch the necessary PyTorch source file. The patch introduces hooks into LocalElasticAgent to initialize the supervisor client and send worker status updates.

1. Create a file named local_elastic_agent.patch.

2. Copy and paste the following content into the file. This patch adds three main functionalities:

   - _setup_goodput_client: Initializes the supervisor client for each worker.

   -  _update_gclient_state: Translates PyTorch WorkerState to supervisor DeviceState.

   - Hooks into _start_workers and _monitor_workers to manage the client and report state changes.

```diff
@@ -15,7 +15,7 @@
 import time
 import uuid
 from string import Template
-from typing import Any, Optional, TYPE_CHECKING
+from typing import Any, Optional, TYPE_CHECKING, Dict

 import torch.distributed.elastic.timer as timer
 from torch.distributed.elastic import events
@@ -162,6 +162,47 @@
         self._worker_watchdog: Optional[timer.FileTimerServer] = None
         self._logs_specs = logs_specs
         self._health_check_server: Optional[HealthCheckServer] = None
+        self._rank_to_gclient: Dict[int, Any] = dict()
+
+    def _setup_goodput_client(self, envs: Dict[int, Dict[str, str]]) -> None:
+
+        for worker_env in envs.values():
+            if worker_env.get('GCP_HOST_DAEMON_PORT') is None:
+                logger.info(f"Not set up goodput client as GCP_HOST_DAEMON_PORT is not set.")
+                return
+            try:
+                rank = int(worker_env["RANK"])
+                local_rank = int(worker_env["LOCAL_RANK"])
+                host_port = int(worker_env["GCP_HOST_DAEMON_PORT"])
+                worker_port = host_port + local_rank + 1
+                import supervisor
+
+                self._rank_to_gclient[rank] = supervisor.GoogleCloudResiliencyClient(
+                    worker_port,
+                    host_port,
+                    local_rank,
+                    rank)
+                logger.info(f"setup_goodput_client at {rank=} success")
+            except ImportError as e:
+                logger.info(f"setup_goodput_client at {rank=} fail")
+                self._rank_to_gclient[rank] = None
+
+    def _update_gclient_state(self, global_rank, worker_state):
+        """
+        Uses pytorch WorkerState to update gclient state.
+        """
+        if self._rank_to_gclient is None or self._rank_to_gclient.get(global_rank) is None:
+            return
+        from supervisor import DeviceState
+        worker_state_to_device_state = {
+            WorkerState.FAILED: DeviceState.FAILED,
+            WorkerState.HEALTHY: DeviceState.RUNNING,
+            WorkerState.SUCCEEDED: DeviceState.COMPLETE,
+        }
+        assert worker_state in worker_state_to_device_state
+        self._rank_to_gclient[global_rank].update_state(worker_state_to_device_state[worker_state])
+
+

     def _setup_local_watchdog(self, envs: dict[int, dict[str, str]]) -> None:
         enable_watchdog_env_name = TORCHELASTIC_ENABLE_FILE_TIMER
@@ -342,6 +383,7 @@

         self._setup_local_watchdog(envs=envs)
         self._setup_healthcheck()
+        self._setup_goodput_client(envs=envs)

         assert spec.entrypoint is not None
         assert self._logs_specs is not None
@@ -353,7 +395,6 @@
             logs_specs=self._logs_specs,
             log_line_prefixes=log_line_prefixes,
             start_method=self._start_method,
-            numa_options=spec.numa_options,
         )

         return self._pcontext.pids()
@@ -394,6 +435,7 @@
                 for local_rank, failure in result.failures.items():
                     worker = worker_group.workers[local_rank]
                     worker_failures[worker.global_rank] = failure
+                    self._update_gclient_state(worker.global_rank, WorkerState.FAILED)
                 return RunResult(
                     state=WorkerState.FAILED,
                     failures=worker_failures,
@@ -404,9 +446,12 @@
                 for local_rank, ret_val in result.return_values.items():
                     worker = worker_group.workers[local_rank]
                     workers_ret_vals[worker.global_rank] = ret_val
+                    self._update_gclient_state(worker.global_rank, WorkerState.SUCCEEDED)
                 return RunResult(
                     state=WorkerState.SUCCEEDED,
                     return_values=workers_ret_vals,
                 )
         else:
+            for global_rank in self._rank_to_gclient.keys():
+                self._update_gclient_state(global_rank, WorkerState.HEALTHY)
             return RunResult(state=WorkerState.HEALTHY)
```

## Step 2: Build a Custom Docker Image
Next, modify your Dockerfile to apply the patch during the image build process. This ensures that your PyTorch environment includes the necessary changes.

Add the following lines to your Dockerfile. Make sure the local_elastic_agent.patch file is in the same directory as your Dockerfile or adjust the COPY path accordingly.

```Dockerfile

# First, find the location of the PyTorch file to be patched.
# This command finds the site-packages directory and constructs the full path.
ARG TORCH_AGENT_PATH=$(python -c "from distutils.sysconfig import get_python_lib; import os; print(os.path.join(get_python_lib(), 'torch/distributed/elastic/agent/server/local_elastic_agent.py'))")

# Copy the patch file into the build context
COPY local_elastic_agent.patch /tmp/local_elastic_agent.patch

# Apply the patch using the `patch` command
RUN patch ${TORCH_AGENT_PATH} < /tmp/local_elastic_agent.patch

# Install gcp resiliency library
RUN pip install google-cloud-resiliency-supervisor
```

Note: This example assumes a Python 3.10 environment. The path to local_elastic_agent.py might differ slightly based on your Python version or base image. The ARG command dynamically finds the correct path.

## Step 3: Run Your PyTorch Workload
With the patched Docker image, you can now run your training job. To activate the supervisor client, you must set the `GCP_HOST_DAEMON_PORT` environment variable when launching your workload. If `GCP_HOST_DAEMON_PORT`, the supervisor client will not be initated by default.

The client code uses this port to connect to the `supervisor` daemon running on the same node.

Set the environment variable in your Kubernetes Pod specification or job manifest:

```YAML

apiVersion: v1
kind: Pod
metadata:
  name: pytorch-training-pod
spec:
  containers:
  - name: pytorch-worker
    image: your-custom-pytorch-image:latest
    env:
    - name: GCP_HOST_DAEMON_PORT
      value: "YOUR_DAEMON_PORT" # e.g., "50051"
    command: ["torchrun", ...]
    args: ["--nproc_per_node=8", "your_script.py"]
```

When the training job starts, the patched code will detect the environment variable, initialize the client, and begin sending worker health status to the supervisor service.

## Step 4: Verify the Integration
After launching your job, you can verify that the integration is working by:

1. Checking Worker Logs: Look for the log message `setup_goodput_client for rank X successful. in your worker pods` logs.

2. Monitoring supervisor: Use the supervisor tools or dashboards to see if it is receiving heartbeats and state updates from your training workers.

3. Simulating Failures: To test the resiliency features, you can simulate a GPU Xid error on one of your nodes. supervisor should detect the failure report from the client, mark the device as unhealthy, and (if configured) trigger a hot-swap to replace the faulty node without terminating the entire job.
