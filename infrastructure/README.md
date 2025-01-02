## Infra setup
### Node provisioning
Use the Terraform code in the *terraform* folder to provision EC2 instances that can serve as a sandbox Kubernetes cluster.
```bash
# AWS api key/secret
export AWS_ACCESS_KEY_ID="YOUR_KEY"
export AWS_SECRET_ACCESS_KEY="YOUR_SECRET"

cd infrastructure/terraform/

# change the "region" variable to suit your preference; default is 'eu-central-1'
# if you change the region, variable "ami_id" must also be adjusted according to your region
vim variables.tf 

terraform init
terraform validate
terraform plan

# create the cloud resources; Beware (!) that you must have already generated a Key pair 'k3s-machines' via the AWS console: EC2 > Networking > Key pairs
terraform apply --auto-approve

# destroy the infrastructure when necessary
terraform destroy --auto-approve
```

### SSH access
```bash
vim /etc/ssh/ssh_config

# Add Hostnames declared below
# configuration for EC2 instances; Ubuntu (!) machines
Host *amazonaws.com
  User ubuntu
  IdentityFile ~/.ssh/k3s-machines.pem # place your private key (generated via the AWS console) in the .ssh/ sub-directory
  IdentitiesOnly yes
  CheckHostIP no
```

### Cluster setup

Initial single-node cluster.
```bash
cd infrastracture/

export HOSTNAME="EC2_MASTER_HOSTNAME" # e.g. ec2-3-68-220-36.eu-central-1.compute.amazonaws.com
export USER=ubuntu
export K3S_VERSION=v1.30.8+k3s1 # same version as the rancher local cluster
export SSH_KEY_PATH="/home/<USER>/.ssh/k3s-machines.pem" # point to the ssh key location locally

# first install the k3sup CLI; traefik not deployed, we use nginx
k3sup install \
  --host $HOSTNAME \
  --user $USER \
  --ssh-key $SSH_KEY_PATH \
  --cluster \
  --k3s-version $K3S_VERSION \
  --k3s-extra-args '--disable traefik'

export KUBECONFIG=${PWD}/kubeconfig

# confirm kubectl functioning
kubectl get node -o wide

# confirm pod scheduling
kubectl run tmp --image=nginx:latest -it --rm --restart=Never -- curl https://go.dev
```

Add *n* more nodes for redundancy; note: workloads can be scheduled on *master* nodes.
```bash
export SERVER_HOST_ONE="EC2_MASTER_HOSTNAME" # existing master node (k3s 'server')
export NEXT_HOST_TWO="EC2_MASTER_HOSTNAME_2" # another 'master' node to be added
export NEXT_HOST_THREE="EC2_WORKER_HOSTNAME_1" # another 'worker' node to be added
...
export USER=ubuntu
export K3S_VERSION=v1.30.8+k3s1
export SSH_KEY_PATH="/home/<USER>/.ssh/k3s-machines.pem" # modify

#  --server flag: Join the cluster as a server (master) rather than as an agent (worker) for the embedded etcd mode; omit if joining a worker-only node
k3sup join \
  --host $NEXT_HOST_TWO \
  --user $USER \
  --server-user $USER \
  --server-host $SERVER_HOST_ONE \
  --server \
  --ssh-key $SSH_KEY_PATH \
  --k3s-version $K3S_VERSION \
  --k3s-extra-args '--disable traefik'

# repeat to join another node; this time as an 'agent' (worker) (--server flag omitted)
k3sup join \
  --host $NEXT_HOST_THREE \
  --user $USER \
  --server-user $USER \
  --server-host $SERVER_HOST_ONE \
  --ssh-key $SSH_KEY_PATH \
  --k3s-version $K3S_VERSION \
  --k3s-extra-args '--disable traefik'
```

Repeat the aforementioned procedure (`k3sup join`) for as many nodes as needed.

## GitOps
### Flux setup
Generate a GitHub PAT that can create repositories by checking all permissions under repo. Then export as an environment variable.

```bash
export GITHUB_TOKEN=<GH_PAT>
```

Run the bootstrap for a repository on your personal GitHub account:
```bash
flux bootstrap github \
  --token-auth \
  --owner=MY-GH-USERNAME \
  --repository=fastapi-image-detection \
  --branch=main \
  --path=infrastructure/gitops \
  --personal

```