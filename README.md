# Multi-Node Distributed Training Cluster using DeepSpeed (CPU-Only)

A production-ready framework executing **Distributed Data Parallelism (DDP)** across a 3-node CPU architecture utilizing **Microsoft DeepSpeed**. This implementation avoids heavy GPU reliance by targeting resource-constrained environments using the PyTorch `Gloo` communication primitive for network gradient synchronization.

## 🏗️ System Architecture

The setup establishes a high-performance distributed cluster comprising one Master node and two Worker nodes running over a private local network subnet (`10.13.0.x`).

* **Node 0 (Master):** `10.13.0.177` (Coordinates runtime environment orchestration and processes dataset shard 0)
* **Node 1 (Worker):** `10.13.0.178` (Processes dataset shard 1)
* **Node 2 (Worker):** `10.13.0.179` (Processes dataset shard 2)

During backpropagation, DeepSpeed aggregates individual forward-pass losses and initiates an **All-Reduce** network collective operation over `pdsh`. This averages gradients globally across the topology before triggering synchronized weights updates.

## 🚀 Key Architectural Overcomes

1. **Absolute Workspace Uniformity:** Engineered matching, user-independent system workspaces (`/opt/project`) globally across nodes to bypass default `pdsh` remote directory synchronization limitations.
2. **Device Isolation Enforcement:** Replaced fallback GPU kernel calls within standard PyTorch DDP scripts with target `cpu` routing, adjusting standard DeepSpeed launcher constraints for zero-GPU clusters.
3. **Ambiguity Resolution:** Streamlined script entry points to allow DeepSpeed to autonomously ingest configuration parameters directly from CLI parsers, eliminating manual runtime dictionary instantiation failures.

## 🛠️ Getting Started

### Prerequisites

All instances must run **Ubuntu 22.04 LTS**, share matching Python environments (`3.10+`), and support passwordless SSH authentication keys from the Master node.

```bash
# Install system runtime engines on all machines
sudo apt-get update && sudo apt-get install -y pdsh ninja-build python3-pip
pip3 install torch deepspeed
