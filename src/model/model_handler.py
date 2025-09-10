#!/usr/bin/env python3

"""
Model Handler for Distributed Inference

This module handles loading, distributing, and running inference on a Small Language Model (SLM)
across multiple nodes in the Ray cluster.
"""

import os
import sys
import time
import yaml
from pathlib import Path
from typing import Dict, List, Any, Optional, Union

import ray
import torch
import numpy as np
from loguru import logger
from transformers import AutoModelForCausalLM, AutoTokenizer

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))


class ModelConfig:
    """Configuration for the language model"""
    
    def __init__(self, config_dict: Dict[str, Any]):
        self.model_name = config_dict.get("model_name", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
        self.quantization = config_dict.get("quantization", "int8")
        self.batch_size = config_dict.get("batch_size", 1)
        self.max_length = config_dict.get("max_length", 512)
        self.temperature = config_dict.get("temperature", 0.7)
        self.top_p = config_dict.get("top_p", 0.9)
        self.device = config_dict.get("device", "cpu")
        
    def __str__(self) -> str:
        return (
            f"ModelConfig(model_name={self.model_name}, "
            f"quantization={self.quantization}, "
            f"batch_size={self.batch_size}, "
            f"max_length={self.max_length}, "
            f"temperature={self.temperature}, "
            f"top_p={self.top_p}, "
            f"device={self.device})"
        )


@ray.remote
class ModelShard:
    """A shard of the model that runs on a single node"""
    
    def __init__(self, config: ModelConfig, shard_id: int, total_shards: int):
        self.config = config
        self.shard_id = shard_id
        self.total_shards = total_shards
        self.model = None
        self.tokenizer = None
        self.device = torch.device(config.device)
        self.load_model()
        
    def load_model(self) -> None:
        """Load the model and tokenizer"""
        logger.info(f"Loading model shard {self.shard_id}/{self.total_shards} on {self.device}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        
        # Load model with quantization if specified
        if self.config.quantization == "int8":
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model_name,
                device_map=self.device,
                load_in_8bit=True,
            )
        elif self.config.quantization == "int4":
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model_name,
                device_map=self.device,
                load_in_4bit=True,
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model_name,
                device_map=self.device,
            )
        
        logger.info(f"Model shard {self.shard_id} loaded successfully")
        
    def generate_text(self, prompt: str) -> str:
        """Generate text based on the prompt"""
        logger.debug(f"Shard {self.shard_id} generating text for prompt: {prompt[:50]}...")
        
        # Tokenize the prompt
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        # Generate text
        with torch.no_grad():
            outputs = self.model.generate(
                inputs["input_ids"],
                max_length=self.config.max_length,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                do_sample=True,
            )
        
        # Decode the generated text
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        return generated_text
    
    def process_batch(self, prompts: List[str]) -> List[str]:
        """Process a batch of prompts"""
        return [self.generate_text(prompt) for prompt in prompts]
    
    def get_shard_info(self) -> Dict[str, Any]:
        """Get information about this model shard"""
        return {
            "shard_id": self.shard_id,
            "total_shards": self.total_shards,
            "model_name": self.config.model_name,
            "device": str(self.device),
            "quantization": self.config.quantization,
        }


class DistributedModel:
    """Manages distributed inference across multiple model shards"""
    
    def __init__(self, config_path: str):
        self.config_path = config_path
        self.model_config = None
        self.model_shards = []
        self.load_config()
        
    def load_config(self) -> None:
        """Load model configuration from YAML"""
        try:
            with open(self.config_path, "r") as f:
                config = yaml.safe_load(f)
            
            # Extract model configuration
            model_config_dict = config.get("model_config", {})
            self.model_config = ModelConfig(model_config_dict)
            
            logger.info(f"Loaded model configuration: {self.model_config}")
        except Exception as e:
            logger.error(f"Failed to load model configuration: {e}")
            raise
    
    def initialize_shards(self) -> None:
        """Initialize model shards across available nodes"""
        if not ray.is_initialized():
            logger.error("Ray is not initialized. Please start Ray before initializing model shards.")
            return
        
        # Get available nodes in the cluster
        nodes = ray.nodes()
        worker_nodes = [node for node in nodes if not node.get("is_head", False)]
        
        if not worker_nodes:
            logger.warning("No worker nodes found. Running on head node only.")
            total_shards = 1
        else:
            logger.info(f"Found {len(worker_nodes)} worker nodes")
            total_shards = len(worker_nodes) + 1  # Include head node
        
        # Create model shards
        logger.info(f"Initializing {total_shards} model shards")
        self.model_shards = [
            ModelShard.remote(self.model_config, i, total_shards)
            for i in range(total_shards)
        ]
        
        # Verify all shards are initialized
        shard_infos = ray.get([shard.get_shard_info.remote() for shard in self.model_shards])
        for info in shard_infos:
            logger.info(f"Shard {info['shard_id']}/{info['total_shards']} initialized on {info['device']}")
    
    def generate_text(self, prompt: str) -> str:
        """Generate text using the distributed model"""
        if not self.model_shards:
            logger.error("Model shards not initialized")
            return "Error: Model not initialized"
        
        # For simplicity, use the first shard for single prompt generation
        # In a real system, you'd implement load balancing
        result = ray.get(self.model_shards[0].generate_text.remote(prompt))
        return result
    
    def generate_batch(self, prompts: List[str]) -> List[str]:
        """Generate text for a batch of prompts, distributed across shards"""
        if not self.model_shards:
            logger.error("Model shards not initialized")
            return ["Error: Model not initialized"] * len(prompts)
        
        # Distribute prompts across shards
        num_shards = len(self.model_shards)
        shard_batches = [[] for _ in range(num_shards)]
        
        # Simple round-robin distribution
        for i, prompt in enumerate(prompts):
            shard_idx = i % num_shards
            shard_batches[shard_idx].append(prompt)
        
        # Process batches in parallel
        batch_results = []
        for i, batch in enumerate(shard_batches):
            if batch:  # Only process non-empty batches
                batch_results.append(self.model_shards[i].process_batch.remote(batch))
        
        # Collect results
        results = ray.get(batch_results)
        
        # Flatten results and reorder to match input order
        flat_results = []
        for i in range(len(prompts)):
            shard_idx = i % num_shards
            batch_idx = i // num_shards
            if batch_idx < len(results[shard_idx]):
                flat_results.append(results[shard_idx][batch_idx])
            else:
                flat_results.append("Error: Failed to generate text")
        
        return flat_results


def main():
    """Test the distributed model"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test the distributed model")
    parser.add_argument(
        "--config", 
        type=str, 
        default="../../config/master.yaml",
        help="Path to the master configuration file"
    )
    args = parser.parse_args()
    
    # Setup logging
    logger.remove()  # Remove default handler
    logger.add(
        sys.stdout,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level="INFO",
    )
    
    # Ensure Ray is initialized
    if not ray.is_initialized():
        ray.init()
    
    # Create distributed model
    model = DistributedModel(args.config)
    model.initialize_shards()
    
    # Test with a single prompt
    prompt = "What is distributed computing?"
    logger.info(f"Testing with prompt: {prompt}")
    
    start_time = time.time()
    result = model.generate_text(prompt)
    end_time = time.time()
    
    logger.info(f"Generated text in {end_time - start_time:.2f} seconds")
    logger.info(f"Result: {result}")
    
    # Test with batch processing
    prompts = [
        "Explain quantum computing in simple terms.",
        "What are the benefits of edge computing?",
        "How does distributed AI differ from traditional AI?",
    ]
    
    logger.info(f"Testing batch processing with {len(prompts)} prompts")
    
    start_time = time.time()
    results = model.generate_batch(prompts)
    end_time = time.time()
    
    logger.info(f"Generated {len(results)} responses in {end_time - start_time:.2f} seconds")
    for i, result in enumerate(results):
        logger.info(f"Result {i+1}: {result[:100]}...")


if __name__ == "__main__":
    main()