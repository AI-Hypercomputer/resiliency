# Goodput Analysis for the job

This tool calculates the following metrics:

- *Total Events*: the total events during the training session
- *Job Started Count*: the number of times the job have started
- *Checkpoints Loaded*: the number of times checkpoints have been loaded
- *Checkpoints Saved*: the number of times a checkpoint have been saved
- *Total Runtime (hours)*: the total number of hours of the job
- *Min Loaded Step*: the minimum step used to load a checkpoint
- *Max Saved Step*: the maximum step on which a checkpoint was saved
- *Step Difference*: the progress made in number of steps
- *Effective Computation Time (hours)*: the amount of time effectively running the training job
- *Goodput Percentage*: the percentage of the total time that the job was making progress

To run the analysis:

1. Create virtual environment and install the required libraries:
```bash
cd $REPO_ROOT/resiliency/goodput_measure
python3 -m venv venv
source venv/bin/activate
python3 -m pip install --upgrade pip
pip3 install -r requirements.txt
```
2. Run the [`calculator.py`](calculator.py)
   script, specifying the <JOBSET_NAME>

```bash
python3 calculator.py --job-name <JOBSET_NAME> \
  --export mymetrics.json \
  --gcloud-logging-lookback-days 1 \
  --verbose \
  --reference-step-time=8
```

To get the <JOBSET_NAME> you can run:
```
kubectl get jobsets
```