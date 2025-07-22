# Set up a cluster with cpu nodes and tpu nodes
# Use headless mode from pathways cloud
https://cloud.google.com/ai-hypercomputer/docs/workloads/pathways-on-cloud/pathways-interactive-mode
* Step 0 (optional, but highly recommended), create your own network and subnet:
The default network is often insufficient or poorly organized for production GKE clusters, especially those requiring large IP ranges for pods/services (like TPU clusters). It's generally a bad practice to use the default network.
```
gcloud compute networks subnets create $SUBNET \
    --network=$VPC \
    --region=us-central1 \
    --range=10.10.0.0/16 \
    --secondary-range pods=10.20.0.0/16,services=10.21.0.0/16 \
    --project=$PROJECT
gcloud compute networks subnets update $SUBNET \
    --region=us-central1 \
    --enable-private-ip-google-access \
    --project=$PROJECT
```
* Step 1, create cluster with TPUs:
```
xpk cluster create-pathways --cluster=swe-test-tpu  --tpu-type=v6e-8     --num-slices=1     --project=${PROJECT}     --zone=${ZONE} --spot --custom-cluster-arguments="--network=swe-vpc --subnetwork=swe-subnet"
```
* Step 2, create headless pathways
```
xpk workload create-pathways \
--headless \
--workload=${WORKLOAD} \
--num-slices=${WORKLOAD_NODEPOOL_COUNT} \
--tpu-type=${TPU_TYPE} \
--project=${PROJECT} \
--zone=${ZONE} \
--cluster=${CLUSTER}
```
* Step 3, port forwarding:
```
PROXY_POD=$(kubectl get pods | grep ${WORKLOAD}-pathways-head | awk '{print $1}')
PROXY_PORT=29000
kubectl port-forward ${PROXY_POD} ${PROXY_PORT}:${PROXY_PORT}
```

* Step 4, in your venv:

pip install pathwaysutils jax

JAX_PLATFORMS=proxy JAX_BACKEND_TARGET=grpc://127.0.0.1:29000 python -c 'import pathwaysutils; import jax; import pprint; pathwaysutils.initialize(); pprint.pprint(jax.devices())'