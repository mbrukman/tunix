"""Sampling client for distributed GRPO training."""

from typing import Any, Dict, List, Optional

from flax import nnx
from jax.typing import ArrayLike

import grpc

from tunix.rl.distributed_learning.config import DistributedLearningConfig
from tunix.rl.distributed_learning.proto import worker_pb2, worker_pb2_grpc
from tunix.rl.distributed_learning.types import ArrayType, DeviceArrayPayload
from tunix.rl.distributed_learning.workers.rollout_worker import RolloutWorker

class RolloutClient:
    """Client for distributed sampling."""

    def __init__(
        self,
        config: DistributedLearningConfig,
        server_address: str,
    ):
        """Initialize the sampling client.

        Args:
            actor_model: The model to use for sampling.
            use_vllm: Whether to use vLLM for faster inference.
            server_address: Address of the sampling worker server.
            collocate: If True, creates a local SampleWorker instead of using RPC.
        """
        self.config = config
        self.collocate = config.train_rollout_collocate
        
        if self.collocate:
            # Create local worker
            self.worker = RolloutWorker(
                model_path=config.train_model_path,
                checkpoint_path=config.train_checkpoint_path,
                host="0.0.0.0",
                port=50051,
                max_worker_threads=config.train_max_worker_threads,
            )
        else:
            # Set up gRPC channel and stub
            self.channel = grpc.insecure_channel(server_address)
            self.worker = worker_pb2_grpc.RolloutWorkerStub(self.channel)

    def generate(
        self,
        prompts: ArrayLike,
        position: ArrayLike,
        max_tokens: int,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: Optional[int] = None,
    ) -> ArrayLike:
        """Generate completions for given prompts.

        Args:
            prompts: Input token IDs as JAX/NumPy array.
            max_tokens: Maximum number of tokens to generate.
            temperature: Sampling temperature.
            top_p: Top-p sampling parameter.
            top_k: Top-k sampling parameter.

        Returns:
            Generated token IDs as JAX array.
        """
        if self.collocate:
            # Use local worker directly
            return self.worker.generate(
                prompts=prompts,
                position=position,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k if top_k is not None else 0,
            )
        else:
            # Convert prompts to DeviceArrayPayload and then to proto
            prompts_payload = DeviceArrayPayload.from_array(prompts)
            prompts_proto = prompts_payload.to_proto()
            
            # Use RPC
            request = worker_pb2.GenerateRequest(
                prompts=prompts_proto,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k if top_k is not None else 0,
            )
            response = self.worker.Generate(request)
            
            # Convert response to JAX array using DeviceArrayPayload
            completions_payload = DeviceArrayPayload.from_proto(response.completions)
            return completions_payload.to_jax()


    def __del__(self):
        """Clean up resources."""
        if hasattr(self, 'channel'):
            self.channel.close()