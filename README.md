# ZanSoc Beta - Distributed AI Compute System

A prototype for a globally distributed AI compute system using low-cost devices. This project demonstrates running a Small Language Model (SLM) across multiple Raspberry Pi nodes connected over the internet using Ray Cluster.

## Overview

This prototype connects devices across different locations (within a ~25 km radius) to distribute AI inference tasks and run a model collaboratively on multiple nodes. It's an experiment in creating a mini decentralized/distributed AI cluster where small devices contribute to powering an AI model together.

## Architecture

- **Master Node**: A laptop or server that coordinates the cluster
- **Worker Nodes**: Three Raspberry Pi devices placed in different locations
- **Connectivity**: Devices connected over WAN/DDNS to form a unified compute cluster
- **Framework**: Ray Cluster for distributed computing
- **Workload**: Small Language Model (SLM) inference distributed across nodes

## Project Structure

```
├── config/                  # Configuration files for Ray Cluster
├── src/                     # Source code
│   ├── master/              # Master node code
│   ├── worker/              # Worker node code
│   └── model/               # SLM model handling code
├── scripts/                 # Utility scripts
├── demo/                    # Demo application
└── requirements.txt         # Python dependencies
```

## Getting Started

See the [setup instructions](./docs/setup.md) for details on how to configure and run the system.

## Goals

- Connect devices across different locations
- Distribute AI inference tasks across them
- Run a model collaboratively on multiple nodes

This is the first step toward a vision of scaling distributed AI beyond just local clusters — moving closer to a global compute fabric where any device can join in.