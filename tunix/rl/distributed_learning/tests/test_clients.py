"""Tests for distributed learning clients."""

import unittest
from unittest.mock import MagicMock, patch

import jax
import jax.numpy as jnp
import optax

from tunix.rl.distributed_learning.clients.train_client import TrainClient
from tunix.rl.distributed_learning.clients.rollout_client import RolloutClient
from tunix.rl.distributed_learning.config import DistributedLearningConfig
from tunix.rl.distributed_learning.types import TrainExample


class TestTrainClient(unittest.TestCase):
    """Tests for TrainClient."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = DistributedLearningConfig(
            train_rollout_collocate=True,
            train_model_path="test_model",
            train_checkpoint_path="test_checkpoint",
            train_max_worker_threads=2,
            total_generation_steps=10,
            max_prompt_length=100,
            lora_enabled=False,
            rollout_model_path="test_model",
            rollout_checkpoint_path="test_checkpoint",
        )
        self.optimizer = optax.adam(1e-4)

    @patch('tunix.rl.distributed_learning.clients.train_client.TrainWorker')
    @patch('tunix.rl.distributed_learning.workers.train_worker.TrainWorker.load_model')
    def test_init_collocated(self, mock_load_model, mock_worker_class):
        """Test initialization in collocated mode."""
        # Mock the worker instance and its load_model method
        mock_worker = MagicMock()
        mock_worker_class.return_value = mock_worker
        mock_load_model.return_value = MagicMock()

        client = TrainClient(
            optimizer=self.optimizer,
            config=self.config,
        )
        self.assertTrue(client.collocate)
        self.assertIsNotNone(client.worker)
        self.assertIsNone(getattr(client, 'channel', None))
        mock_worker_class.assert_called_once()

    @patch('grpc.insecure_channel')
    def test_init_remote(self, mock_channel):
        """Test initialization in remote mode."""
        config = DistributedLearningConfig(
            train_rollout_collocate=False,
            train_model_path="test_model",
            train_checkpoint_path="test_checkpoint",
            train_max_worker_threads=2,
            total_generation_steps=10,
            max_prompt_length=100,
            lora_enabled=False,
            rollout_model_path="test_model",
            rollout_checkpoint_path="test_checkpoint",
        )
        mock_channel.return_value = MagicMock()
        
        client = TrainClient(
            optimizer=self.optimizer,
            config=config,
            server_address="localhost:50051",
        )
        self.assertFalse(client.collocate)
        self.assertIsNotNone(client.channel)
        self.assertIsNotNone(client.worker)
        mock_channel.assert_called_once_with("localhost:50051")

    @patch('tunix.rl.distributed_learning.clients.train_client.TrainWorker')
    @patch('tunix.rl.distributed_learning.workers.train_worker.TrainWorker.load_model')
    def test_train_step_collocated(self, mock_load_model, mock_worker_class):
        """Test train_step in collocated mode."""
        # Mock the worker instance and its methods
        mock_worker = MagicMock()
        mock_worker.train_step.return_value = (0.5, {"loss": 0.5})  # Mock train_step to return (loss, aux)
        mock_worker_class.return_value = mock_worker
        mock_load_model.return_value = MagicMock()

        client = TrainClient(
            optimizer=self.optimizer,
            config=self.config,
        )
        
        # Create a test example
        example = TrainExample(
            prompt_ids=jnp.zeros((1, 3)),
            prompt_mask=jnp.zeros((1, 3)),
            completion_ids=jnp.zeros((1, 3)),
            completion_mask=jnp.zeros((1, 3)),
            advantages=jnp.zeros((1,)),
            ref_per_token_logps=jnp.zeros((1, 3)),
            old_per_token_logps=jnp.zeros((1, 3))
        )
        
        # Test train_step
        loss, aux = client.train_step(example)
        self.assertEqual(loss, 0.5)
        self.assertEqual(aux["loss"], 0.5)
        mock_worker.train_step.assert_called_once_with(example)

    @patch('grpc.insecure_channel')
    def test_train_step_remote(self, mock_channel):
        """Test train_step in remote mode."""
        config = DistributedLearningConfig(
            train_rollout_collocate=False,
            train_model_path="test_model",
            train_checkpoint_path="test_checkpoint",
            train_max_worker_threads=2,
            total_generation_steps=10,
            max_prompt_length=100,
            lora_enabled=False,
            rollout_model_path="test_model",
            rollout_checkpoint_path="test_checkpoint",
        )
        
        # Mock the gRPC channel and stub
        mock_stub = MagicMock()
        mock_stub.Train.return_value = MagicMock(loss=0.5, aux={"loss": 0.5})
        mock_channel.return_value = MagicMock()
        
        client = TrainClient(
            optimizer=self.optimizer,
            config=config,
            server_address="localhost:50051",
        )
        client.worker = mock_stub
        
        # Create a mock TrainExample with all 7 fields
        example = TrainExample(
            prompt_ids=jnp.zeros((1, 3)),
            prompt_mask=jnp.zeros((1, 3)),
            completion_ids=jnp.zeros((1, 3)),
            completion_mask=jnp.zeros((1, 3)),
            advantages=jnp.zeros((1,)),
            ref_per_token_logps=jnp.zeros((1, 3)),
            old_per_token_logps=jnp.zeros((1, 3))
        )
        
        loss, aux = client.train_step(example)
        self.assertEqual(loss, 0.5)
        self.assertEqual(aux["loss"], 0.5)
        mock_stub.Train.assert_called_once()


class TestRolloutClient(unittest.TestCase):
    """Tests for RolloutClient."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = DistributedLearningConfig(
            train_rollout_collocate=True,
            rollout_model_path="test_model",
            rollout_checkpoint_path="test_checkpoint",
            rollout_max_worker_threads=2,
            total_generation_steps=10,
            max_prompt_length=100,
            lora_enabled=False,
            train_model_path="test_model",
            train_checkpoint_path="test_checkpoint",
        )

    @patch('tunix.rl.distributed_learning.clients.rollout_client.RolloutWorker')
    @patch('tunix.rl.distributed_learning.workers.rollout_worker.RolloutWorker.load_model')
    @patch('tunix.rl.distributed_learning.workers.rollout_worker.RolloutWorker.generate')
    def test_init_collocated(self, mock_generate, mock_load_model, mock_worker_class):
        """Test initialization in collocated mode."""
        # Mock the worker instance and its load_model method
        mock_worker = MagicMock()
        mock_worker_class.return_value = mock_worker
        mock_load_model.return_value = MagicMock()
        mock_generate.return_value = MagicMock()

        client = RolloutClient(
            config=self.config,
            server_address="localhost:50051",
        )
        self.assertTrue(client.collocate)
        self.assertIsNotNone(client.worker)
        self.assertIsNone(getattr(client, 'channel', None))
        mock_worker_class.assert_called_once()

    @patch('grpc.insecure_channel')
    def test_init_remote(self, mock_channel):
        """Test initialization in remote mode."""
        config = DistributedLearningConfig(
            train_rollout_collocate=False,
            train_model_path="test_model",
            train_checkpoint_path="test_checkpoint",
            train_max_worker_threads=2,
            total_generation_steps=10,
            max_prompt_length=100,
            lora_enabled=False,
            rollout_model_path="test_model",
            rollout_checkpoint_path="test_checkpoint",
        )
        mock_channel.return_value = MagicMock()
        
        client = RolloutClient(
            config=config,
            server_address="localhost:50051",
        )
        self.assertFalse(client.collocate)
        self.assertIsNotNone(client.channel)
        self.assertIsNotNone(client.worker)
        mock_channel.assert_called_once_with("localhost:50051")

    @patch('tunix.rl.distributed_learning.clients.rollout_client.RolloutWorker')
    @patch('tunix.rl.distributed_learning.workers.rollout_worker.RolloutWorker.load_model')
    def test_generate_collocated(self, mock_load_model, mock_worker_class):
        """Test generate in collocated mode."""
        # Mock the worker instance and its methods
        mock_worker = MagicMock()
        mock_worker.generate.return_value = jnp.zeros((1, 3))  # Mocked completion array
        mock_worker_class.return_value = mock_worker
        mock_load_model.return_value = MagicMock()

        client = RolloutClient(
            config=self.config,
            server_address="localhost:50051",
        )
        
        prompts = jnp.array([[1, 2, 3]])
        position = jnp.array([[3]])
        completions = client.generate(
            prompts=prompts,
            position=position,
            max_tokens=10,
            temperature=0.7,
            top_p=0.9,
            top_k=50,
        )
        self.assertEqual(completions.shape, (1, 3))
        mock_worker.generate.assert_called_once_with(
            prompts=prompts,
            position=position,
            max_tokens=10,
            temperature=0.7,
            top_p=0.9,
            top_k=50,
        )

if __name__ == '__main__':
    unittest.main() 