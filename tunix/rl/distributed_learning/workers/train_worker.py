"""Training worker implementation for distributed GRPO training.

This worker handles the actual training computation, either running locally or
serving requests through gRPC. It manages model training, evaluation, and weight
synchronization with the rollout worker.
"""

from typing import Any, Dict, List, Optional, Tuple, TypeVar

import jax
import jax.numpy as jnp
from flax import nnx
import optax
import grpc
from concurrent import futures

from tunix.rl import common
from tunix.rl.distributed_learning.workers.base_worker import Worker
from tunix.rl.distributed_learning.config import DistributedLearningConfig
from tunix.rl.grpo.grpo_helpers import compute_advantages, compute_kl_divergence
from tunix.rl.distributed_learning.proto import worker_pb2, worker_pb2_grpc
from tunix.rl.distributed_learning.types import ArrayType, DeviceArrayPayload, TrainExample
from tunix.rl.distributed_learning.server import WorkerServer

from typing_extensions import override

class TrainWorkerServicer(worker_pb2_grpc.TrainWorkerServicer):
    """gRPC implementation for training worker.

    This servicer handles incoming gRPC requests for training operations,
    converting between proto messages and JAX arrays.
    """

    def __init__(self, worker: 'TrainWorker'):
        """Initialize the implementation.

        Args:
            worker: The training worker instance that handles the actual computation.
        """
        self.worker = worker

    def TrainStep(
        self,
        request: worker_pb2.TrainRequest,
        context: grpc.ServicerContext,
    ) -> worker_pb2.TrainResponse:
        """Handle train request.

        Args:
            request: The train request containing the batch data.
            context: gRPC context.

        Returns:
            Train response containing loss and auxiliary metrics.
        """
        # Convert each field from proto to JAX array
        train_example = TrainExample(
            prompt_ids=DeviceArrayPayload.from_proto(request.prompt_ids).to_jax(),
            prompt_mask=DeviceArrayPayload.from_proto(request.prompt_mask).to_jax(),
            completion_ids=DeviceArrayPayload.from_proto(request.completion_ids).to_jax(),
            completion_mask=DeviceArrayPayload.from_proto(request.completion_mask).to_jax(),
            advantages=DeviceArrayPayload.from_proto(request.advantages).to_jax(),
            ref_per_token_logps=DeviceArrayPayload.from_proto(request.ref_per_token_logps).to_jax(),
            old_per_token_logps=DeviceArrayPayload.from_proto(request.old_per_token_logps).to_jax()
        )
        
        # Train step
        loss, aux = self.worker.train_step(train_example)
        
        return worker_pb2.TrainResponse(
            loss=float(loss),
            aux=aux,
        )

class TrainWorker(Worker):
    """Worker for distributed training.

    This worker handles the actual training computation, including model training,
    evaluation, and weight synchronization with the rollout worker.
    """

    @override
    def load_model(self, model_path: str, checkpoint_path: str) -> nnx.Module:
        """Load model from checkpoint.

        Args:
            model_path: Path to model definition.
            checkpoint_path: Path to model checkpoint.

        Returns:
            Loaded model instance.
        """
        # TODO: Implement model loading
        raise NotImplementedError("Model loading not implemented yet")
    
    def train_step(
        self,
        example: TrainExample
    ) -> Tuple[float, Dict[str, float]]:
        """Execute a training step.

        Args:
            example: Training example containing prompts, completions,
                advantages, and log probabilities.

        Returns:
            Tuple of (loss, auxiliary metrics).
        """
        # TODO: Implement train step
        raise NotImplementedError("Training step not implemented yet")

    def get_metrics(self) -> Dict[str, float]:
        """Get current training metrics.

        Returns:
            Dictionary of metric names to values.
        """
        # TODO: Implement get metrics
        raise NotImplementedError("Get metrics not implemented yet")

    def sync_weights(self) -> None:
        """Reshard and sync weights from training model to sampling model.

        This method takes the training model weights, reshards them for the sampling
        model's device mesh, and updates the sampling model's weights.
        """
        raise NotImplementedError("Sync weights not implemented yet")
    
    def get_per_token_logps(
        self,
        prompts: jax.Array,
        completions: jax.Array
    ) -> Tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        """Get per-token log probabilities from reference and policy models.

        Args:
            prompts: Input token IDs.
            completions: Generated token IDs.

        Returns:
            Tuple of (reference model log probabilities, policy model log probabilities,
            completion mask, prompt-completion mask).
        """
        raise NotImplementedError("Get per token logps not implemented yet")
    
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
        raise NotImplementedError("Eval not implemented yet")

    def start_server(self) -> None:
        """Start the gRPC server.

        This initializes and starts the gRPC server with the TrainWorkerServicer
        to handle incoming training requests.
        """
        self.server = WorkerServer(
            worker_type="TrainWorker",
            host=self.host,
            port=self.port,
            max_workers=self.max_worker_threads,
        )

        self.server.start(TrainWorkerServicer(self))
        self.server.wait_for_termination()

    def __del__(self):
        """Clean up resources."""
        if hasattr(self, 'server') and self.server:
            self.server.stop()
