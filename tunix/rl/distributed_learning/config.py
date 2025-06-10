"""Base configuration classes for distributed learning."""

from dataclasses import dataclass
from typing import Optional


@dataclass(slots=True, kw_only=True)
class DistributedLearningConfig:
    """Base configuration for distributed learning.
    
    This class defines the common configuration parameters needed for distributed
    learning, which can be extended by specific algorithms like GRPO and DPO.
    """
    # Training parameters
    learning_rate: float = 1e-5
    max_grad_norm: float = 1.0
    weight_decay: float = 0.01
    
    # Model parameters
    model_name: str = "gpt2"
    model_revision: str = "main"
    model_dtype: str = "float32"
    
    # Distributed training parameters
    num_workers: int = 1
    batch_size: int = 8
    gradient_accumulation_steps: int = 1
    
    # Logging and evaluation
    logging_steps: int = 10
    eval_steps: int = 100
    save_steps: int = 1000
    
    # Optional parameters
    seed: Optional[int] = None

    train_rollout_collocate: bool

    # GRPO Trainer Config
    total_generation_steps: int
    num_generations: int = 2
    num_iterations: int = 1
    beta: float = 0.04
    epsilon: float = 0.2
    temperature: float = 0.9
    top_p: float = 1.0
    top_k: int | None = None
    max_prompt_length: int

    lora_enabled: bool

    # Worker options, put together with other options for convenience of creating collocated instance
    # Rollout worker options
    rollout_model_path: str
    rollout_checkpoint_path: str
    rollout_max_worker_threads: int = 10

    # Train worker options
    train_model_path: str
    train_checkpoint_path: str
    train_max_worker_threads: int = 10

