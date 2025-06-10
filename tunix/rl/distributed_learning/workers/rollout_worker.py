"""Rollout worker implementation for distributed GRPO training."""

from typing import Any, Dict, List, Optional, TypeVar

import jax
import jax.numpy as jnp
from flax import nnx
import grpc
from concurrent import futures

from tunix.rl.distributed_learning.workers.base_worker import Worker
from tunix.rl.distributed_learning.proto import worker_pb2, worker_pb2_grpc
from tunix.rl.distributed_learning.types import ArrayType, DeviceArrayPayload
from tunix.rl.distributed_learning.server import WorkerServer


class RolloutWorkerServicer(worker_pb2_grpc.RolloutWorkerServicer):
    """gRPC implementation for rollout worker."""

    def __init__(self, worker: 'RolloutWorker'):
        """Initialize the implementation.

        Args:
            worker: The rollout worker instance.
        """
        self.worker = worker

    def Generate(
        self,
        request: worker_pb2.GenerateRequest,
        context: grpc.ServicerContext,
    ) -> worker_pb2.GenerateResponse:
        """Handle generate request.

        Args:
            request: The generate request.
            context: gRPC context.

        Returns:
            Generate response.
        """
        # Convert proto to DeviceArrayPayload and then to JAX array
        prompts_payload = DeviceArrayPayload.from_proto(request.prompts)
        prompts = prompts_payload.to_jax()
        
        # Create position IDs
        position = jnp.arange(len(prompts))
        
        # Generate completions
        completions = self.worker.generate(
            prompts=prompts,
            position=position,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k,
        )
        
        # Convert completions to DeviceArrayPayload and then to proto
        completions_payload = DeviceArrayPayload.from_array(completions)
        completions_proto = completions_payload.to_proto()
        
        return worker_pb2.GenerateResponse(completions=completions_proto)


class RolloutWorker(Worker):
    """Worker for distributed rollout."""

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

    def load_model(self, model_path: str, checkpoint_path: str) -> nnx.Module:
        """Load model from checkpoint.

        Args:
            model_path: Path to model.
            checkpoint_path: Path to model checkpoint.
        Returns:
            Loaded model.
        """
        # TODO: Implement model loading, maybe async
        raise NotImplementedError("Model loading not implemented yet")
    
    def generate(
        self,
        prompts: jax.Array,
        position: jax.Array,
        max_tokens: int = 100,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = 0,
    ) -> jax.Array:
        """Generate completions for given prompts.

        Args:
            prompts: Input token IDs.
            position: Position IDs.
            max_tokens: Maximum number of tokens to generate.
            temperature: Sampling temperature.
            top_p: Top-p sampling parameter.
            top_k: Top-k sampling parameter.

        Returns:
            Generated token IDs.
        """
        # TODO: Implement generate
        raise NotImplementedError("Rollout generation not implemented yet")

    def start_server(self) -> None:
        """Start the gRPC server."""
        if self.server is not None:
            raise RuntimeError("Server is already running")
            
        self.server = WorkerServer(
            worker_type="RolloutWorker",
            host=self.host,
            port=self.port,
            max_workers=self.max_worker_threads,
        )
        self.server.start(RolloutWorkerServicer(self))
        self.server.wait_for_termination()

    def __del__(self):
        """Clean up resources."""
        if hasattr(self, 'server') and self.server:
            self.server.stop() 