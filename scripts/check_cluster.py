#!/usr/bin/env python3

"""
Cluster Status Checker

This script checks the status of the Ray cluster and displays information about connected nodes.
"""

import os
import sys
import yaml
import argparse
from pathlib import Path
from typing import Dict, List, Any

import ray
from loguru import logger

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))


def setup_logger():
    """Configure the logger"""
    logger.remove()  # Remove default handler
    logger.add(
        sys.stdout,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level="INFO",
    )


def load_config(config_path):
    """Load the configuration from YAML"""
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        sys.exit(1)


def check_cluster_status(config):
    """Check the status of the Ray cluster"""
    try:
        # Check if Ray is already initialized
        if not ray.is_initialized():
            # Extract configuration values
            master_ip = config.get("master_ip", "127.0.0.1")
            master_port = config.get("port", 6379)
            redis_password = config.get("redis_password", None)
            
            # Connect to the cluster
            address = f"{master_ip}:{master_port}"
            logger.info(f"Connecting to Ray cluster at {address}")
            
            ray.init(
                address=address,
                _redis_password=redis_password,
            )
        
        # Get cluster info
        nodes = ray.nodes()
        
        if not nodes:
            logger.warning("No nodes found in the cluster")
            return
        
        # Display cluster information
        logger.info(f"Ray Cluster Status: {len(nodes)} nodes connected")
        logger.info("="*50)
        
        # Find head node
        head_nodes = [node for node in nodes if node.get("is_head", False)]
        worker_nodes = [node for node in nodes if not node.get("is_head", False)]
        
        if head_nodes:
            head_node = head_nodes[0]
            logger.info("Head Node:")
            logger.info(f"  - Node ID: {head_node['node_id']}")
            logger.info(f"  - Address: {head_node['node_ip_address']}")
            logger.info(f"  - Resources: {head_node['resources']}")
            logger.info("")
        else:
            logger.warning("No head node found")
        
        if worker_nodes:
            logger.info(f"Worker Nodes ({len(worker_nodes)}):\n")
            for i, node in enumerate(worker_nodes):
                logger.info(f"Worker {i+1}:")
                logger.info(f"  - Node ID: {node['node_id']}")
                logger.info(f"  - Address: {node['node_ip_address']}")
                logger.info(f"  - Resources: {node['resources']}")
                logger.info("")
        else:
            logger.warning("No worker nodes found")
        
        # Display cluster resources
        total_resources = {}
        for node in nodes:
            for resource, value in node.get("resources", {}).items():
                if resource in total_resources:
                    total_resources[resource] += value
                else:
                    total_resources[resource] = value
        
        logger.info("Total Cluster Resources:")
        for resource, value in total_resources.items():
            logger.info(f"  - {resource}: {value}")
        
        return nodes
    except Exception as e:
        logger.error(f"Failed to check cluster status: {e}")
        return None


def main():
    """Main entry point"""
    # Set up command line arguments
    parser = argparse.ArgumentParser(description="Check the status of the ZanSoc Ray cluster")
    parser.add_argument(
        "--config", 
        type=str, 
        default="../config/master.yaml",
        help="Path to the configuration file"
    )
    args = parser.parse_args()
    
    # Setup logging
    setup_logger()
    
    # Load configuration
    config_path = args.config
    if not os.path.isabs(config_path):
        # Convert to absolute path relative to the script location
        config_path = os.path.join(os.path.dirname(__file__), config_path)
    
    config = load_config(config_path)
    
    # Check cluster status
    nodes = check_cluster_status(config)
    
    # Shutdown Ray
    if ray.is_initialized():
        ray.shutdown()
        logger.info("Disconnected from Ray cluster")
    
    if nodes:
        return 0
    else:
        return 1


if __name__ == "__main__":
    sys.exit(main())