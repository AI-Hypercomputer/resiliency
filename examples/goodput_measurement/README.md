# Goodput Measurement

This guide provides a step-by-step tutorial for integrating the `resiliency.goodput_measure` library into a PyTorch workload. By following these instructions, you can capture the necessary job events to calculate the goodput of your training job.

## Goodput Primer

Goodput is a measure of the useful work accomplished by a system over a given period. In the context of model training, it represents the percentage of time spent on effective computation, excluding time lost to failures, restarts, and other overheads.

To illustrate this, consider the following example event log for a training job:

| Event Type | Timestamp (minutes) | Notes |
| :--- | :--- | :--- |
| Job Started | 0 | The clock starts. |
| Step=25 Checkpoint Loaded | 5 | Loading pre-existing work from step=25. |
| Step=50 Finished | 55 | 25 steps of progress made, regular speed 2min/step. |
| Step=50 Checkpoint Saved | 60 | Take 5min to save ckpt, progress is now safe. |
| Step=70 Finished | 100 | 20 more steps of progress. |
| Job Crashed | 100 | Failure! Work after step 50 is lost. |
| Job Restarted | 115 | System is recovering. |
| Step=50 Checkpoint Loaded | 120 | Restarting from the last good state. |
| Step=75 Finished | 195 | Finished another 25 steps, but training has straggler, it is 3min/step. |
| Step=75 Checkpoint Saved | 200 | Final progress is now safe. |
| Job Finished | 200 | The clock stops. |

### Measuring Goodput

Based on the event log, we can calculate the following:

*   **Total Time:** The job ran from the start (0 min) to the final finish (200 min).
    `200 - 0 = 200 minutes`
*   **Effective Computation Time:** This is the ideal time required for the useful work accomplished.
    *   **Useful Work:** Net progress from Step=25 to Step=75 = 50 steps.
    *   **Ideal Speed:** The baseline performance was 2 minutes/step.
    *   **Effective Time:** `50 steps × 2 min/step = 100 minutes`
*   **Final Goodput:**
    `(100 Effective Mins / 200 Total Mins) x 100% = 50%`

### Measuring Badput

"Badput" represents the time that did not contribute to effective computation. Breaking it down helps identify areas for improvement.

*   **Training Progress Loss (40 mins / 20%):** Work that was completed but lost due to the crash. It's the time from the last successful save (@60) to the crash (@100).
    `100 - 60 = 40 minutes`
*   **Training Slowdown (25 mins / 12.5%):** The second run took 75 minutes but should have ideally taken 50 minutes (25 steps * 2 min/step). The difference is time lost to inefficiency.
    `75 (actual) - 50 (ideal) = 25 minutes`
*   **Job Restart Overhead (15 mins / 7.5%):** Time the system was down between the crash (@100) and the job restarting (@115).
    `115 - 100 = 15 minutes`
*   **Checkpointing Overhead (20 mins / 10%):** Total time spent on non-computation tasks of saving and loading data.
    `Loading: 10 mins + Saving: 10 mins = 20 minutes`

### Summary

| Category | Time (min) | Percentage |
| :--- | :--- | :--- |
| **Total Time** | **200** | **100%** |
| Effective Computation Time | 100 | 50% |
| Checkpoint Loading | 10 | 5% |
| Training Progress Loss | 40 | 20% |
| Job Restart | 15 | 7.5% |
| Checkpoint Saving | 10 | 5% |
| Training Slowdown | 25 | 12.5% |

## Capturing Job Events

To calculate goodput, you need to log key events during the training job's lifecycle.

### Capture Job Start and Termination Events

These events can be captured in the `torchrun` launcher script. The following diff shows how to add logging for `JOB_STARTED` and `JOB_TERMINATED` events:

```diff
--- a/third_party/nvidia-resiliency-ext/v0.4.1/launcher.py
+++ b/third_party/nvidia-resiliency-ext/v0.4.1/launcher.py
@@ -88,6 +88,9 @@
     write_obj_to_ipc_stream,
 )

+from resiliency.goodput_measure import constant as goodput_event
+from resiliency.goodput_measure import logging as goodput_logging
+
 logging.basicConfig(
     level=os.getenv('FT_LAUNCHER_LOGLEVEL', 'INFO'),
     format=f"[%(asctime)s] [%(levelname)s] [ft_launcher{os.getpid()}@{socket.gethostname()}] %(message)s",
@@ -1059,6 +1062,10 @@
         restart_policy=config.restart_policy,
         is_store_host=_is_store_host(rdzv_parameters),
     )
+    goodput_logging.log_event(
+      event_type=goodput_event.JOB_STARTED,
+      node_rank=int(os.environ.get("NODE_RANK")),
+    )

     shutdown_rdzv = True
     try:
@@ -1110,6 +1117,10 @@
         raise
     finally:
         agent.clean_rdzv_shutdown(close=shutdown_rdzv)
+        goodput_logging.log_event(
+            event_type=goodput_event.JOB_TERMINATED,
+            node_rank=int(os.environ.get("NODE_RANK")),
+        )
         agent.shutdown_rank_monitors()
         with contextlib.suppress(Exception):
             os.unlink(FT_LAUNCHER_IPC_SOCKET)

```

### Capture Checkpoint Saving and Loading Events

These events should be logged when a checkpoint is saved or loaded. The current step info should also be logged to analyze the training progress. The following diff for `fsdp_example.py` illustrates how to capture these events:

```diff
--- a/examples/goodput_measurement/fsdp_example.py
+++ b/examples/goodput_measurement/fsdp_example.py
@@ -11,6 +11,9 @@ from torch.distributed.checkpoint.stateful import Stateful
 from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict
 from torch.distributed.fsdp import fully_shard

+from resiliency.goodput_measure import logging as goodput_logging
+from resiliency.goodput_measure import constant as goodput_event
+
 # --- Configuration ---
 CHECKPOINT_DIR = "my_fsdp_checkpoint"
 MAX_STEPS = 10000
@@ -127,6 +130,9 @@ def run_training():
         if rank == 0:
             print("No valid checkpoint subfolders found. Starting training from scratch.")

+    if rank == 0:
+        goodput_logging.log_event(goodput_event.CHECKPOINT_LOADED, step=start_step)
+
     # 4. Training Loop
     last_time, last_step = time.time(), start_step
     for step in range(start_step, MAX_STEPS):
@@ -160,6 +166,7 @@ def run_training():
             )

             if rank == 0:
+                goodput_logging.log_event(goodput_event.CHECKPOINT_SAVED, step=app_state.step)
                 print("Checkpoint saved successfully.\n")

     cleanup()
```

## Calculating Goodput from Event Logs

The library allows you to log events to a local file or to Google Cloud Logging. Once the events are logged, you can use the `goodput_measure/calculator.py` utility to calculate the goodput.

Here is a concrete example using `fsdp_example.py`. First, run the training job:

```bash
GOODPUT_USE_FILE_TRACKING=true python3 /app/resiliency/third_party/nvidia-resiliency-ext/v0.4.1/launcher.py \
--nproc-per-node="${GPUS_PER_NODE}" \
                --nnodes="${NNODES}" \
                --node-rank="${NODE_RANK}" \
                --rdzv-id="${JOB_IDENTIFIER}" \
                --rdzv-endpoint="${MASTER_ADDR}:${MASTER_PORT}"
```

During the run, we can interrupt the job with `Ctrl+C` and restart it. This will generate an event log similar to the following:

```json
{"timestamp": "2025-09-27T04:51:00.492242", "job_name": "yejingxin-dev2-goodput-n1--a3h-unittest-09261818", "event_type": "job_started", "node_rank": 0}
{"timestamp": "2025-09-27T04:51:29.376985", "job_name": "yejingxin-dev2-goodput-n1--a3h-unittest-09261818", "event_type": "checkpoint_loaded", "step": 0}
{"timestamp": "2025-09-27T04:51:35.775347", "job_name": "yejingxin-dev2-goodput-n1--a3h-unittest-09261818", "event_type": "checkpoint_saved", "step": 500}
{"timestamp": "2025-09-27T04:51:37.153714", "job_name": "yejingxin-dev2-goodput-n1--a3h-unittest-09261818", "event_type": "checkpoint_saved", "step": 1000}
{"timestamp": "2025-09-27T04:51:38.505679", "job_name": "yejingxin-dev2-goodput-n1--a3h-unittest-09261818", "event_type": "checkpoint_saved", "step": 1500}
{"timestamp": "2025-09-27T04:51:39.863866", "job_name": "yejingxin-dev2-goodput-n1--a3h-unittest-09261818", "event_type": "checkpoint_saved", "step": 2000}
{"timestamp": "2025-09-27T04:51:41.196246", "job_name": "yejingxin-dev2-goodput-n1--a3h-unittest-09261818", "event_type": "checkpoint_saved", "step": 2500}
{"timestamp": "2025-09-27T04:51:42.528565", "job_name": "yejingxin-dev2-goodput-n1--a3h-unittest-09261818", "event_type": "checkpoint_saved", "step": 3000}
{"timestamp": "2025-09-27T04:51:43.860838", "job_name": "yejingxin-dev2-goodput-n1--a3h-unittest-09261818", "event_type": "checkpoint_saved", "step": 3500}
{"timestamp": "2025-09-27T04:51:45.199876", "job_name": "yejingxin-dev2-goodput-n1--a3h-unittest-09261818", "event_type": "checkpoint_saved", "step": 4000}
{"timestamp": "2025-09-27T04:51:47.533941", "job_name": "yejingxin-dev2-goodput-n1--a3h-unittest-09261818", "event_type": "job_terminated", "node_rank": 0}
{"timestamp": "2025-09-27T04:52:07.013451", "job_name": "yejingxin-dev2-goodput-n1--a3h-unittest-09261818", "event_type": "job_started", "node_rank": 0}
{"timestamp": "2025-09-27T04:52:40.627857", "job_name": "yejingxin-dev2-goodput-n1--a3h-unittest-09261818", "event_type": "checkpoint_loaded", "step": 4000}
{"timestamp": "2025-09-27T04:52:42.189783", "job_name": "yejingxin-dev2-goodput-n1--a3h-unittest-09261818", "event_type": "checkpoint_saved", "step": 4500}
{"timestamp": "2025-09-27T04:52:43.491595", "job_name": "yejingxin-dev2-goodput-n1--a3h-unittest-09261818", "event_type": "checkpoint_saved", "step": 5000}
{"timestamp": "2025-09-27T04:52:44.810060", "job_name": "yejingxin-dev2-goodput-n1--a3h-unittest-09261818", "event_type": "checkpoint_saved", "step": 5500}
{"timestamp": "2025-09-27T04:52:46.185741", "job_name": "yejingxin-dev2-goodput-n1--a3h-unittest-09261818", "event_type": "checkpoint_saved", "step": 6000}
{"timestamp": "2025-09-27T04:52:47.544125", "job_name": "yejingxin-dev2-goodput-n1--a3h-unittest-09261818", "event_type": "checkpoint_saved", "step": 6500}
{"timestamp": "2025-09-27T04:52:48.881336", "job_name": "yejingxin-dev2-goodput-n1--a3h-unittest-09261818", "event_type": "checkpoint_saved", "step": 7000}
{"timestamp": "2025-09-27T04:52:50.207887", "job_name": "yejingxin-dev2-goodput-n1--a3h-unittest-09261818", "event_type": "checkpoint_saved", "step": 7500}
{"timestamp": "2025-09-27T04:52:51.524989", "job_name": "yejingxin-dev2-goodput-n1--a3h-unittest-09261818", "event_type": "checkpoint_saved", "step": 8000}
{"timestamp": "2025-09-27T04:52:52.847352", "job_name": "yejingxin-dev2-goodput-n1--a3h-unittest-09261818", "event_type": "checkpoint_saved", "step": 8500}
{"timestamp": "2025-09-27T04:52:54.193366", "job_name": "yejingxin-dev2-goodput-n1--a3h-unittest-09261818", "event_type": "checkpoint_saved", "step": 9000}
{"timestamp": "2025-09-27T04:52:55.475204", "job_name": "yejingxin-dev2-goodput-n1--a3h-unittest-09261818", "event_type": "checkpoint_saved", "step": 9500}
{"timestamp": "2025-09-27T04:52:56.741096", "job_name": "yejingxin-dev2-goodput-n1--a3h-unittest-09261818", "event_type": "checkpoint_saved", "step": 10000}
{"timestamp": "2025-09-27T04:52:58.746024", "job_name": "yejingxin-dev2-goodput-n1--a3h-unittest-09261818", "event_type": "job_terminated", "node_rank": 0}
```

### Determine the Reference Step Time

The "reference step time" is the ideal time to complete one training step, without considering other operations like checkpointing. This is usually obtained by running a small number of steps without any checkpointing and measuring the average time per step for the last half of the steps (excluding the warmup steps).

Once you have the event log and the reference step time, you can run the calculator to get the goodput analysis:

```bash
python3 resiliency/goodput_measure/calculator.py --job-name=yejingxin-dev2-goodput-n1--a3h-unittest-09261818 --log-file=./yejingxin-dev2-goodput-n1--a3h-unittest-09261818-goodput.log --reference-step-time=0.002569
```

The output will be a summary of the goodput analysis:

```
=== Goodput Analysis for Job: yejingxin-dev2-goodput-n1--a3h-unittest-09261818 ===

+------------------------------------+---------+
| Metric                             | Value   |
+====================================+=========+
| Total Events                       | 27      |
+------------------------------------+---------+
| Job Started Count                  | 2       |
+------------------------------------+---------+
| Checkpoints Loaded                 | 1       |
+------------------------------------+---------+
| Checkpoints Saved                  | 20      |
+------------------------------------+---------+
| Total Runtime (hours)              | 0.03    |
+------------------------------------+---------+
| Min Loaded Step                    | 0       |
+------------------------------------+---------+
| Max Saved Step                     | 10000   |
+------------------------------------+---------+
| Step Difference                    | 10000   |
+------------------------------------+---------+
| Effective Computation Time (hours) | 0.01    |
+------------------------------------+---------+
| Goodput Percentage                 | 21.72%  |
+------------------------------------+---------+
```
