# Weekend Experment

- Create a bug under this [component](https://b.corp.google.com/issues/new?component=1784216&template=0),
the bug should include head commit hash of the repo, the cmd started the exp, any cmd used to restart the exp, and the final goodput analysis report.

Example bug message:
```
head  commit 4338d9a32b067037fc142eee286148ab7e4b10b9
helm install -f helm/supervisor-chart/values/weekend-exp.yaml $USER helm/supervisor-chart
helm install -f helm/workload-chart/values/weekend-exp.yaml $USER-exp-0509 helm/workload-chart
```

- Checkout to repo head and build the latest image

```
git checkout origin/main
cd docker
bash build_image.sh
```

- Test Run on Friday Morning
    - Start supervisor
    ```
    helm install -f helm/supervisor-chart/values/weekend-exp.yaml $USER helm/supervisor-chart
    ```
    - Get job date tag
    ```
    export JOB_TAG=$(date +%m%d)

    ```
    If you need to manually restart the same experiment on a different day, please **do not modify the job tag**. Continue using the original tag to maintain a consistent job name across runs in the same experiment. This ensures accurate experiment analysis and allows checkpoint loading to function correctly.

    - Start workload
    ```
    helm install -f helm/workload-chart/values/weekend-exp.yaml $USER-exp-$JOB_TAG helm/workload-chart \
    --set workload.model_size=36M \
    --set infrastructure.num_nodes=2 \
    --set workload.num_opt_replicas=2 \
    --set workload.additional_flag=--sim-fault-desc='random\,120'
    ```
  Make sure job can finish with 200 steps without any issue

- Start Real run on Weekend Signup slot
    - Start supervisor
    ```
    helm install -f helm/supervisor-chart/values/weekend-exp.yaml $USER helm/supervisor-chart
    ```

    - Start workloads
    ```
    helm install -f helm/workload-chart/values/weekend-exp.yaml $USER-exp-$JOB_TAG-w1 helm/workload-chart \
    --set workload.model_size=405B \
    --set infrastructure.num_nodes=36 \
    --set workload.num_opt_replicas=2 \
    --set workload.supervisor.enable_workload_scaling=false \
    --set fault_injection.enable_google_xid_injection=true \
    --set fault_injection.google_fault_injection_interval_s=7200 \
    --set workload.additional_flag=--sim-fault-desc='random\,3600'

    helm install -f helm/workload-chart/values/weekend-exp.yaml $USER-exp-$JOB_TAG-w2 helm/workload-chart \
    --set workload.model_size=70B \
    --set infrastructure.num_nodes=8 \
    --set workload.num_opt_replicas=2 \
    --set fault_injection.enable_google_xid_injection=true \
    --set fault_injection.google_fault_injection_interval_s=7200 \
    --set workload.additional_flag=--sim-fault-desc='random\,3600'
    ```
