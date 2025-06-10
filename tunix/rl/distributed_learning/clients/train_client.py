"""Training client for distributed GRPO training.

This client handles communication with the training worker, either through gRPC or
directly when collocated. It manages training steps, evaluation, and model state
synchronization.
"""

from typing import Any, Dict, List, Optional, Tuple

from flax import nnx
import optax
import grpc
import jax

from tunix.rl.distributed_learning.config import DistributedLearningConfig
from tunix.rl.distributed_learning.workers.train_worker import TrainWorker
from tunix.rl.distributed_learning.proto import worker_pb2, worker_pb2_grpc
from tunix.rl.distributed_learning.types import ArrayType, DeviceArrayPayload, TrainExample


class TrainClient:
    """Client for distributed training.

    This client handles communication with the training worker, either through gRPC
    or directly when collocated. It manages training steps, evaluation, and model
    state synchronization.
    """

    def __init__(
        self,
        optimizer: optax.GradientTransformation,
        config: DistributedLearningConfig,
        server_address: str = "localhost:50051",
    ):
        """Initialize the training client.

        Args:
            optimizer: The optimizer to use for training.
            config: Training configuration including collocation settings.
            server_address: Address of the training worker server. Only used when
                not collocated.
        """
        self.optimizer = optimizer
        self.config = config
        self.collocate = config.train_rollout_collocate
        
        if self.collocate:
            # Create local worker
            self.worker = TrainWorker(
                model_path=config.train_model_path,
                checkpoint_path=config.train_checkpoint_path,
                host="0.0.0.0",
                port=50051,
                max_worker_threads=config.train_max_worker_threads,
            )
        else:
            # Set up gRPC channel and stub
            self.channel = grpc.insecure_channel(server_address)
            self.worker = worker_pb2_grpc.TrainWorkerStub(self.channel)

    def train_step(
        self,
        train_example: TrainExample
    ) -> Tuple[float, Dict[str, float]]:
        """Execute a training step.

        Args:
            train_example: Training example containing prompts, completions,
                advantages, and log probabilities.

        Returns:
            Tuple of (loss, auxiliary metrics).
        """
        if self.collocate:
            # Use local worker directly
            return self.worker.train_step(train_example)
        else:
            # Convert each field of TrainExample to DeviceArrayPayload and then to proto
            request = worker_pb2.TrainRequest(
                prompt_ids=DeviceArrayPayload.from_array(train_example.prompt_ids).to_proto(),
                prompt_mask=DeviceArrayPayload.from_array(train_example.prompt_mask).to_proto(),
                completion_ids=DeviceArrayPayload.from_array(train_example.completion_ids).to_proto(),
                completion_mask=DeviceArrayPayload.from_array(train_example.completion_mask).to_proto(),
                advantages=DeviceArrayPayload.from_array(train_example.advantages).to_proto(),
                ref_per_token_logps=DeviceArrayPayload.from_array(train_example.ref_per_token_logps).to_proto(),
                old_per_token_logps=DeviceArrayPayload.from_array(train_example.old_per_token_logps).to_proto()
            )
            
            # Use RPC
            response = self.worker.Train(request)
            
            return response.loss, response.aux

    def get_per_token_logps(self, prompts: jax.Array, completions: jax.Array) -> Tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        """Get per-token log probabilities from reference and policy models.

        Args:
            prompts: Input token IDs.
            completions: Generated token IDs.

        Returns:
            Tuple of (reference model log probabilities, policy model log probabilities,
            completion mask, prompt-completion mask).
        """
        raise NotImplementedError("Get per token logps not implemented yet")
    
    def sync_weights(self) -> None:
        """Sync model weights with the rollout worker.

        This ensures the rollout worker has the latest model parameters for
        generation.
        """
        raise NotImplementedError("Sync weights not implemented yet")

    def maybe_restore(self) -> int:
        """Restore model parameters from the latest checkpoint if available.

        Returns:
            Number of training steps completed.
        """
        raise NotImplementedError("Maybe restore not implemented yet")

    def eval(self, example: TrainExample) -> Tuple[float, Dict[str, float]]:
        """Run evaluation on the given example.

        Args:
            example: Evaluation example containing prompts, completions,
                advantages, and log probabilities.

        Returns:
            Tuple of (loss, auxiliary metrics).
        """
        if self.collocate:
            # Use local worker directly
            return self.worker.eval(**example)
        else:
            # Convert each field of TrainExample to DeviceArrayPayload and then to proto
            request = worker_pb2.EvalRequest(
                prompt_ids=DeviceArrayPayload.from_array(example.prompt_ids).to_proto(),
                prompt_mask=DeviceArrayPayload.from_array(example.prompt_mask).to_proto(),
                completion_ids=DeviceArrayPayload.from_array(example.completion_ids).to_proto(),
                completion_mask=DeviceArrayPayload.from_array(example.completion_mask).to_proto(),
                advantages=DeviceArrayPayload.from_array(example.advantages).to_proto(),
                ref_per_token_logps=DeviceArrayPayload.from_array(example.ref_per_token_logps).to_proto(),
                old_per_token_logps=DeviceArrayPayload.from_array(example.old_per_token_logps).to_proto()
            )
            
            # Use RPC
            response = self.worker.Eval(request)
            
            return response.loss, response.aux

    def __del__(self):
        """Clean up resources."""
        if hasattr(self, 'channel'):
            self.channel.close() 