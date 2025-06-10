"""Base server implementation for distributed learning workers."""

from typing import Any, Optional
import grpc
from concurrent import futures

from tunix.rl.distributed_learning.proto import worker_pb2_grpc


class WorkerServer:
    """Base class for worker servers."""

    def __init__(
        self,
        worker_type: str,
        host: str,
        port: int,
        max_workers: int,
    ):
        """Initialize the worker server.

        Args:
            worker_type: Type of worker (e.g., "RolloutWorker", "TrainWorker", "SampleWorker").
            host: Host address to bind to. Defaults to "0.0.0.0".
            port: Port to listen on. Defaults to 50051.
            max_workers: Maximum number of worker threads. Defaults to 10.
        """
        self.worker_type = worker_type
        self.host = host
        self.port = port
        self.max_workers = max_workers
        self.server = None

    def start(self, servicer: Any) -> None:
        """Start the gRPC server.

        Args:
            servicer: The gRPC servicer instance.
        """
        if self.server is not None:
            raise RuntimeError("Server is already running")

        # Create gRPC server
        self.server = grpc.server(futures.ThreadPoolExecutor(max_workers=self.max_workers))

        # Add servicer based on worker type
        try:
            servicer_func = getattr(worker_pb2_grpc, f"add_{self.worker_type}Servicer_to_server")
            servicer_func(servicer, self.server)
        except AttributeError:
            raise ValueError(f"Unknown worker type: {self.worker_type}")

        # Start server
        self.server.add_insecure_port(f"{self.host}:{self.port}")
        self.server.start()

        print(f"{self.worker_type} server started on {self.host}:{self.port}")

    def stop(self) -> None:
        """Stop the gRPC server."""
        if self.server is not None:
            self.server.stop(0)
            self.server = None
            print(f"{self.worker_type} server stopped")

    def wait_for_termination(self) -> None:
        """Wait for the server to terminate."""
        if self.server is None:
            raise RuntimeError("Server is not running")

        try:
            self.server.wait_for_termination()
        except KeyboardInterrupt:
            print(f"\nShutting down {self.worker_type} server...")
            self.stop() 