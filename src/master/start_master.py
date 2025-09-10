#!/usr/bin/env python3

"""
Master Node Starter Script

This script initializes the Ray head node (master) for the ZanSoc distributed AI compute system.
It reads configuration from the master.yaml file and starts a Ray head node with the specified settings.
"""

import os
import sys
import yaml
import time
import socket
import argparse
from pathlib import Path

import ray
from ray import tune
from loguru import logger

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))


def setup_logger():
    """Configure the logger for the master node"""
    logger.remove()  # Remove default handler
    logger.add(
        sys.stdout,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level="INFO",
    )
    logger.add(
        "logs/master_{time}.log",
        rotation="100 MB",
        retention="10 days",
        level="DEBUG",
    )


def load_config(config_path):
    """Load the master node configuration from YAML"""
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        logger.info(f"Loaded configuration from {config_path}")
        return config
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        sys.exit(1)


def get_public_ip():
    """Get the public-facing IP address of this machine"""
    try:
        # This is a simple way to get the local IP, not the public IP
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception as e:
        logger.warning(f"Could not determine public IP: {e}")
        return "127.0.0.1"  # Fallback to localhost


def start_ray_head(config):
    """Start the Ray head node with the given configuration"""
    try:
        # Extract configuration values
        port = config.get("port", 6379)
        redis_password = config.get("redis_password", None)
        dashboard_host = config.get("dashboard_host", "0.0.0.0")
        dashboard_port = config.get("dashboard_port", 8265)
        resources = config.get("resources", {})
        log_to_driver = config.get("log_to_driver", True)
        logging_level = config.get("logging_level", "info")
        
        # Initialize Ray head node
        ray_init_args = {
            "address": None,  # Start a new Ray cluster
            "num_cpus": resources.get("CPU", 4),
            "dashboard_host": dashboard_host,
            "dashboard_port": dashboard_port,
            "log_to_driver": log_to_driver,
            "logging_level": logging_level,
            "include_dashboard": True,
        }
        
        logger.info("Starting Ray head node with the following configuration:")
        logger.info(f"  - Port: {port}")
        logger.info(f"  - Dashboard: http://{dashboard_host}:{dashboard_port}")
        logger.info(f"  - Resources: {resources}")
        
        # Initialize Ray
        ray.init(**ray_init_args)
        
        # Get cluster info
        nodes_info = ray.nodes()
        logger.info(f"Ray cluster started with {len(nodes_info)} nodes")
        
        # Print connection information for workers
        ip = get_public_ip()
        logger.info(f"\nTo connect worker nodes, use the following address:")
        logger.info(f"  ray start --address='{ip}:{port}' --redis-password='{redis_password}'")
        
        # Keep the script running
        uptime_s = config.get("uptime_s", 86400)  # Default to 24 hours
        logger.info(f"\nMaster node will run for {uptime_s} seconds")
        logger.info("Press Ctrl+C to stop the master node")
        
        # Main loop to keep the script running
        start_time = time.time()
        try:
            while time.time() - start_time < uptime_s:
                time.sleep(10)  # Check every 10 seconds
                # Periodically log cluster status
                if int((time.time() - start_time) % 600) < 10:  # Every 10 minutes
                    nodes = ray.nodes()
                    logger.info(f"Cluster status: {len(nodes)} nodes connected")
        except KeyboardInterrupt:
            logger.info("Received keyboard interrupt, shutting down...")
        
        return True
    except Exception as e:
        logger.error(f"Failed to start Ray head node: {e}")
        return False


def main():
    """Main entry point for the master node starter"""
    # Set up command line arguments
    parser = argparse.ArgumentParser(description="Start the ZanSoc master node")
    parser.add_argument(
        "--config", 
        type=str, 
        default="../../config/master.yaml",
        help="Path to the master configuration file"
    )
    args = parser.parse_args()
    
    # Setup logging
    setup_logger()
    logger.info("Starting ZanSoc master node")
    
    # Ensure logs directory exists
    os.makedirs("logs", exist_ok=True)
    
    # Load configuration
    config_path = args.config
    if not os.path.isabs(config_path):
        # Convert to absolute path relative to the script location
        config_path = os.path.join(os.path.dirname(__file__), config_path)
    
    config = load_config(config_path)
    
    # Start Ray head node
    success = start_ray_head(config)
    
    if success:
        logger.info("Master node stopped successfully")
        return 0
    else:
        logger.error("Failed to start master node")
        return 1


if __name__ == "__main__":
    sys.exit(main())