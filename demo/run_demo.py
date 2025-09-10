#!/usr/bin/env python3

"""
ZanSoc Demo Application

This script demonstrates the distributed AI compute capabilities of the ZanSoc system
by running inference tasks across multiple nodes in the Ray cluster.
"""

import os
import sys
import time
import yaml
import argparse
from pathlib import Path
from typing import List, Dict, Any

import ray
from loguru import logger

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

# Import model handler
from src.model.model_handler import DistributedModel


def setup_logger():
    """Configure the logger"""
    logger.remove()  # Remove default handler
    logger.add(
        sys.stdout,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level="INFO",
    )
    logger.add(
        "logs/demo_{time}.log",
        rotation="100 MB",
        retention="10 days",
        level="DEBUG",
    )


def load_config(config_path):
    """Load the configuration from YAML"""
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        logger.info(f"Loaded configuration from {config_path}")
        return config
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        sys.exit(1)


def connect_to_cluster(config):
    """Connect to the Ray cluster"""
    try:
        # Check if Ray is already initialized
        if ray.is_initialized():
            logger.info("Already connected to Ray cluster")
            return True
        
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
        logger.info(f"Connected to Ray cluster with {len(nodes)} nodes")
        
        return True
    except Exception as e:
        logger.error(f"Failed to connect to Ray cluster: {e}")
        return False


def run_single_inference_demo(model):
    """Run a demo with single inference requests"""
    logger.info("\n=== Single Inference Demo ===")
    
    # Test prompts
    prompts = [
        "Explain how distributed computing works in simple terms.",
        "What are the advantages of edge computing over cloud computing?",
        "How can small devices contribute to AI computation?",
        "What is the future of decentralized AI systems?",
    ]
    
    for i, prompt in enumerate(prompts):
        logger.info(f"\nPrompt {i+1}: {prompt}")
        
        # Measure inference time
        start_time = time.time()
        result = model.generate_text(prompt)
        end_time = time.time()
        
        logger.info(f"Generated response in {end_time - start_time:.2f} seconds")
        logger.info(f"Response: {result}")


def run_batch_inference_demo(model):
    """Run a demo with batch inference requests"""
    logger.info("\n=== Batch Inference Demo ===")
    
    # Test batch prompts
    batch_prompts = [
        "What is Ray Cluster and how does it work?",
        "Explain the concept of distributed AI inference.",
        "How do Small Language Models compare to Large Language Models?",
        "What are the challenges in running AI models on edge devices?",
        "How can Raspberry Pi devices be used for AI computation?",
        "What is the potential of globally distributed compute systems?",
        "How does model sharding work in distributed AI?",
        "What are the bandwidth requirements for distributed inference?",
    ]
    
    logger.info(f"Processing batch of {len(batch_prompts)} prompts")
    
    # Measure batch inference time
    start_time = time.time()
    results = model.generate_batch(batch_prompts)
    end_time = time.time()
    
    total_time = end_time - start_time
    avg_time = total_time / len(batch_prompts)
    
    logger.info(f"Processed {len(batch_prompts)} prompts in {total_time:.2f} seconds")
    logger.info(f"Average time per prompt: {avg_time:.2f} seconds")
    
    # Display results
    for i, (prompt, result) in enumerate(zip(batch_prompts, results)):
        logger.info(f"\nPrompt {i+1}: {prompt}")
        logger.info(f"Response: {result[:100]}...")


def run_stress_test(model, num_prompts=20):
    """Run a stress test with many concurrent requests"""
    logger.info(f"\n=== Stress Test ({num_prompts} prompts) ===")
    
    # Generate test prompts
    base_prompts = [
        "Explain {topic} in simple terms.",
        "What are the advantages of {topic}?",
        "How does {topic} work?",
        "What is the future of {topic}?",
        "Compare {topic} with traditional approaches.",
    ]
    
    topics = [
        "distributed computing",
        "edge AI",
        "federated learning",
        "model parallelism",
        "data parallelism",
        "decentralized systems",
        "mesh networks",
        "swarm intelligence",
        "peer-to-peer computing",
        "IoT networks",
    ]
    
    # Generate prompts by combining base prompts and topics
    prompts = []
    for _ in range(num_prompts):
        base_idx = len(prompts) % len(base_prompts)
        topic_idx = len(prompts) % len(topics)
        prompt = base_prompts[base_idx].format(topic=topics[topic_idx])
        prompts.append(prompt)
    
    logger.info(f"Processing {len(prompts)} prompts in parallel")
    
    # Measure batch inference time
    start_time = time.time()
    results = model.generate_batch(prompts)
    end_time = time.time()
    
    total_time = end_time - start_time
    avg_time = total_time / len(prompts)
    throughput = len(prompts) / total_time
    
    logger.info(f"Processed {len(prompts)} prompts in {total_time:.2f} seconds")
    logger.info(f"Average time per prompt: {avg_time:.2f} seconds")
    logger.info(f"Throughput: {throughput:.2f} prompts/second")
    
    # Display summary of results
    logger.info(f"\nSuccessfully generated {len([r for r in results if not r.startswith('Error')])} responses")
    if any(r.startswith('Error') for r in results):
        logger.warning(f"Failed to generate {len([r for r in results if r.startswith('Error')])} responses")


def main():
    """Main entry point for the demo"""
    # Set up command line arguments
    parser = argparse.ArgumentParser(description="Run the ZanSoc distributed AI demo")
    parser.add_argument(
        "--config", 
        type=str, 
        default="../config/master.yaml",
        help="Path to the configuration file"
    )
    parser.add_argument(
        "--stress-test",
        action="store_true",
        help="Run a stress test with many concurrent requests"
    )
    parser.add_argument(
        "--num-prompts",
        type=int,
        default=20,
        help="Number of prompts to use in the stress test"
    )
    args = parser.parse_args()
    
    # Setup logging
    setup_logger()
    logger.info("Starting ZanSoc Demo")
    
    # Ensure logs directory exists
    os.makedirs("logs", exist_ok=True)
    
    # Load configuration
    config_path = args.config
    if not os.path.isabs(config_path):
        # Convert to absolute path relative to the script location
        config_path = os.path.join(os.path.dirname(__file__), config_path)
    
    config = load_config(config_path)
    
    # Connect to the Ray cluster
    if not connect_to_cluster(config):
        logger.error("Failed to connect to Ray cluster. Exiting.")
        return 1
    
    try:
        # Initialize the distributed model
        logger.info("Initializing distributed model")
        model = DistributedModel(config_path)
        model.initialize_shards()
        
        # Run demos
        run_single_inference_demo(model)
        run_batch_inference_demo(model)
        
        # Run stress test if requested
        if args.stress_test:
            run_stress_test(model, args.num_prompts)
        
        logger.info("\nDemo completed successfully")
        return 0
    except Exception as e:
        logger.error(f"Error running demo: {e}")
        return 1
    finally:
        # Shutdown Ray
        if ray.is_initialized():
            ray.shutdown()
            logger.info("Disconnected from Ray cluster")


if __name__ == "__main__":
    sys.exit(main())