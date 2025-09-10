# Setup Instructions

## Prerequisites

### Master Node
- Python 3.9+ installed
- Public IP address or DDNS configured
- Open ports for Ray Cluster communication (default: 6379, 10001-10999)

### Worker Nodes (Raspberry Pi)
- Raspberry Pi 4 (2GB+ RAM) recommended
- Raspberry Pi OS (64-bit) installed
- Python 3.9+ installed
- Network connectivity to the master node
- Static IP or DDNS configured

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/zansoc-beta.git
cd zansoc-beta
```

### 2. Install Dependencies

On all nodes (master and workers):

```bash
pip install -r requirements.txt
```

## Configuration

### 1. Master Node Setup

1. Edit the `config/master.yaml` file to set your master node's IP/hostname and port

```yaml
# Example master.yaml
cluster_name: "zansoc-cluster"
max_workers: 3
uptime_s: 86400  # 24 hours
redis_password: "your_secure_password"
```

2. Start the Ray head node:

```bash
cd zansoc-beta
python src/master/start_master.py
```

### 2. Worker Node Setup

1. Edit the `config/worker.yaml` file on each worker node to point to your master node:

```yaml
# Example worker.yaml
master_ip: "your-master-ddns.example.com"  # or IP address
master_port: 6379
redis_password: "your_secure_password"
node_resources:
  CPU: 4
  GPU: 0
  memory: 2000000000  # ~2GB in bytes
```

2. Start the Ray worker on each Raspberry Pi:

```bash
cd zansoc-beta
python src/worker/start_worker.py
```

## Verifying the Cluster

On the master node, you can verify that all workers have joined the cluster:

```bash
python scripts/check_cluster.py
```

This should display information about all connected nodes.

## Running the Demo

Once your cluster is set up, you can run the demo application:

```bash
python demo/run_demo.py
```

This will load a Small Language Model and distribute inference tasks across all nodes in the cluster.

## Troubleshooting

### Connection Issues

- Ensure all nodes can reach each other over the network
- Verify that required ports are open on all firewalls
- Check DDNS configuration if using dynamic IPs

### Performance Issues

- Monitor CPU/memory usage on Raspberry Pis
- Adjust batch sizes in the model configuration
- Consider reducing model size if nodes are overloaded

### Ray Cluster Issues

- Check Ray logs in `/tmp/ray/session_latest/logs/`
- Restart nodes if they become unresponsive
- Verify Redis password is consistent across all nodes