#!/usr/bin/env python3

"""
Worker Node Starter Script

This script initializes a Ray worker node for the ZanSoc distributed AI compute system.
It reads configuration from the worker.yaml file and connects to the specified master node.
"""

import os
import sys
import yaml
import time
import socket
import argparse
from pathlib import Path

import ray
from loguru import logger

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))


def setup_logger():
    """Configure the logger for the worker node"""
    logger.remove()  # Remove default handler
    logger.add(
        sys.stdout,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level="INFO",
    )
    logger.add(
        "logs/worker_{time}.log",
        rotation="100 MB",
        retention="10 days",
        level="DEBUG",
    )


def load_config(config_path):
    """Load the worker node configuration from YAML"""
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        logger.info(f"Loaded configuration from {config_path}")
        return config
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        sys.exit(1)


def get_node_name(config):
    """Generate a unique name for this worker node"""
    base_name = config.get("node_name", "raspberry-pi-worker")
    # Use last part of hostname as unique identifier
    hostname = socket.gethostname()
    short_hostname = hostname.split(".")[0]
    return f"{base_name}-{short_hostname}"


def connect_to_master(config):
    """Connect to the Ray head node as a worker"""
    try:
        # Extract configuration values
        master_ip = config.get("master_ip")
        master_port = config.get("master_port", 6379)
        redis_password = config.get("redis_password")
        log_to_driver = config.get("log_to_driver", True)
        logging_level = config.get("logging_level", "info")
        connection_retries = config.get("connection_retries", 5)
        connection_timeout_s = config.get("connection_timeout_s", 30)
        
        if not master_ip:
            logger.error("Master IP address not specified in configuration")
            return False
        
        # Prepare connection address
        address = f"{master_ip}:{master_port}"
        
        logger.info(f"Connecting to Ray head node at {address}")
        
        # Try to connect with retries
        for attempt in range(connection_retries):
            try:
                # Initialize Ray worker - removed resource parameters
                ray.init(
                    address=address,
                    _redis_password=redis_password,
                    log_to_driver=log_to_driver,
                    logging_level=logging_level
                )
                
                logger.info("Successfully connected to Ray head node")
                return True
            except Exception as e:
                logger.warning(f"Connection attempt {attempt+1}/{connection_retries} failed: {e}")
                if attempt < connection_retries - 1:
                    logger.info(f"Retrying in {connection_timeout_s} seconds...")
                    time.sleep(connection_timeout_s)
                else:
                    logger.error("All connection attempts failed")
                    return False
    except Exception as e:
        logger.error(f"Failed to connect to Ray head node: {e}")
        return False


def main():
    """Main entry point for the worker node starter"""
    # Set up command line arguments
    parser = argparse.ArgumentParser(description="Start a ZanSoc worker node")
    parser.add_argument(
        "--config", 
        type=str, 
        default="../../config/worker.yaml",
        help="Path to the worker configuration file"
    )
    args = parser.parse_args()
    
    # Setup logging
    setup_logger()
    logger.info("Starting ZanSoc worker node")
    
    # Ensure logs directory exists
    os.makedirs("logs", exist_ok=True)
    
    # Load configuration
    config_path = args.config
    if not os.path.isabs(config_path):
        # Convert to absolute path relative to the script location
        config_path = os.path.join(os.path.dirname(__file__), config_path)
    
    config = load_config(config_path)
    
    # Connect to master node
    success = connect_to_master(config)
    
    if success:
        logger.info("Worker node connected successfully")
        
        # Keep the worker running
        try:
            logger.info("Worker node is running. Press Ctrl+C to stop.")
            # This will keep the script running until Ray is shut down
            while ray.is_initialized():
                time.sleep(10)  # Check every 10 seconds
        except KeyboardInterrupt:
            logger.info("Received keyboard interrupt, shutting down...")
        finally:
            # Disconnect from Ray
            if ray.is_initialized():
                ray.shutdown()
                logger.info("Disconnected from Ray cluster")
        
        return 0
    else:
        logger.error("Failed to start worker node")
        return 1


if __name__ == "__main__":
    sys.exit(main())