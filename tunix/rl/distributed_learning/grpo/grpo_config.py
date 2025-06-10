"""GRPO-specific configuration."""

from dataclasses import dataclass
from typing import Optional

from tunix.rl.distributed_learning.config import DistributedLearningConfig


@dataclass
class GrpoConfig(DistributedLearningConfig):
    """Configuration for GRPO training.
    
    This class extends the base distributed learning configuration with
    GRPO-specific parameters.
    """
    # GRPO-specific parameters
    beta: float = 0.1  # KL penalty coefficient
    epsilon: float = 0.2  # PPO clip range
    num_generations: int = 4  # Number of generations per prompt
    
    # Optional GRPO parameters
    ref_model_name: Optional[str] = None  # If None, uses same model as actor
    ref_model_revision: str = "main"
    ref_model_dtype: str = "float32" 