"""Base worker class for distributed learning."""

from typing import Any, Dict
from abc import ABC

from flax import nnx


class Worker(ABC):
    """Base worker class for distributed learning."""

    def __init__(
        self,
        model_path: str,
        checkpoint_path: str,
        host: str = "0.0.0.0",
        port: int = 50051,
        max_worker_threads = 10,
    ):
        """Initialize the rollout worker."""
        self.model = self.load_model(model_path, checkpoint_path)
        self.host = host
        self.port = port
        self.max_worker_threads = max_worker_threads
        self.server = None  # Initialize as None

    def load_model(model_path: str, checkpoint_path: str):
        raise NotImplementedError("Model loading not implemented yet")

    def get_model_state(self) -> Dict[str, Any]:
        """Get the current model state.

        Returns:
            Dictionary containing the model state.
        """
        return nnx.state(self.model, nnx.Param) 
    
    