"""GRPO client for distributed learning."""

from typing import Any, Dict, List, Optional, Tuple, Iterable, Sequence, Callable

import flax
import jax
import jax.numpy as jnp
from jax.typing import ArrayLike
from flax import nnx
import optax
import numpy as np
from tqdm import tqdm
import logging
import time
from contextlib import contextmanager

from tunix.rl.distributed_learning.config import DistributedLearningConfig
from tunix.rl.distributed_learning.clients.train_client import TrainClient
from tunix.rl.distributed_learning.clients.rollout_client import RolloutClient
from tunix.rl.distributed_learning.types import ArrayType, DeviceArrayPayload, TrainExample
from tunix.rl.grpo.grpo_helpers import compute_advantages
from tunix.rl.grpo.grpo_trainer import RepeatTrainingInputIter

from tunix.sft.metrics_logger import MetricsLogger, Mode

_TrainingInputT = Dict[str, List[str] | ArrayLike]

@contextmanager
def time_measure(name: str):
    """Context manager for timing code blocks."""
    start = time.time()
    yield
    end = time.time()
    logging.info("%s took %f seconds", name, end - start)


class ProgressBar:
    """Progress bar for training."""

    def __init__(
        self,
        metrics_logger: Any,
        initial_steps: int,
        max_steps: int,
    ):
        """Initialize progress bar.

        Args:
            metrics_logger: Logger for metrics.
            initial_steps: Initial step count.
            max_steps: Maximum number of steps.
        """
        self.metrics_logger = metrics_logger
        self.steps = initial_steps
        self.max_steps = max_steps
        self.pbar = tqdm(total=max_steps - initial_steps)

    def update(self, metrics: Dict[str, float], increment_steps: bool = True):
        """Update progress bar.

        Args:
            metrics: Current metrics.
            increment_steps: Whether to increment step count.
        """
        if increment_steps:
            self.steps += 1
            self.pbar.update(1)
        
        # Update description with metrics
        desc = " | ".join(f"{k}: {v:.4f}" for k, v in metrics.items())
        self.pbar.set_description(desc)

    def close(self):
        """Close progress bar."""
        self.pbar.close()


class MetricsLogger:
    """Logger for training metrics."""

    def __init__(self):
        """Initialize metrics logger."""
        self.metrics = {
            "train": {},
            "eval": {},
        }

    def log_metric(self, name: str, value: float, split: str = "train"):
        """Log a metric.

        Args:
            name: Metric name.
            value: Metric value.
            split: Data split (train/eval).
        """
        if name not in self.metrics[split]:
            self.metrics[split][name] = []
        self.metrics[split][name].append(value)

    def get_metric(self, name: str, split: str = "train") -> float:
        """Get a metric value.

        Args:
            name: Metric name.
            split: Data split (train/eval).

        Returns:
            Latest metric value.
        """
        if name not in self.metrics[split]:
            return 0.0
        return self.metrics[split][name][-1]


class Throttler:
    """Throttles computation to avoid OOM."""

    def __init__(self, max_computations: int = 1):
        """Initialize throttler.

        Args:
            max_computations: Maximum number of concurrent computations.
        """
        self.max_computations = max_computations
        self.current_computations = 0

    def wait_for_next(self):
        """Wait for next computation slot."""
        while self.current_computations >= self.max_computations:
            time.sleep(0.1)

    def add_computation(self, _: Any):
        """Add a computation.

        Args:
            _: Computation result (unused).
        """
        self.current_computations += 1

    def wait_for_all(self):
        """Wait for all computations to complete."""
        while self.current_computations > 0:
            time.sleep(0.1)


class GrpoClient:
    """Client for distributed GRPO learning."""

    def __init__(
        self,
        optimizer: optax.GradientTransformation,
        config: DistributedLearningConfig,
        reward_fns: Callable[..., List[float]] | List[Callable[..., List[float]]],
        train_server_address: str = "localhost:50051",
        rollout_server_address: str = "localhost:50052",
    ):
        """Initialize the GRPO client.

        Args:
            config: Training configuration.
            reward_fns: A single callable or a list of callables that compute a scalar
                reward for given prompts and completions. Each function should accept
                `prompts`, `completions` and optional keyword arguments, and return a
                list of float rewards.
        """
        self.optimizer = optimizer
        self.grpo_config = config
        self.reward_fns = [reward_fns] if not isinstance(reward_fns, Sequence) else reward_fns

        # Initialize clients
        self.train_client = TrainClient(
            optimizer=optimizer,
            config=self.grpo_config,
            server_address=train_server_address,
        )
        self.rollout_client = RolloutClient(
            self.grpo_config,
            server_address=rollout_server_address,
        )

        # Initialize training state
        self._train_steps = 0
        self._metrics_logger = MetricsLogger()
        self._throttler = Throttler()
        self._tqdm_train_metrics = {}
        self._mode: Mode = Mode.TRAIN

    def _tqdm_train_metrics(self) -> list[str]:
        metrics = ["loss", "perplexity", "rewards/overall"]
        if self.grpo_config.beta != 0.0:
            metrics.append("kl")
        return metrics

    def _tqdm_eval_metrics(self) -> list[str]:
        metrics = ["loss", "perplexity", "rewards/overall"]
        if self.grpo_config.beta != 0.0:
            metrics.append("kl")
        return metrics

    def _get_metric_logging_steps(self) -> int:
        return (
            self._train_steps
            if self._mode == self._metrics_logger.Mode.TRAIN
            else self._eval_steps
        )

    def _prepare_inputs(self, training_input: _TrainingInputT) -> Any:
        """Prepare inputs for training.

        Args:
            example: Input example.

        Returns:
            Prepared inputs.
        """
        if self._mode == self._metrics_logger.Mode.TRAIN:
            idx = self._train_steps % len(self._data_buffer)
            data = self._data_buffer[idx]
            if data is None or self._num_iterations == 0:
                data = self._generate_and_compute_advantage(training_input)
                self._data_buffer[idx] = data

            grad_acc_steps = self.grpo_config.get_with_default(
                "gradient_accumulation_steps", 1
            )
            if self._train_steps % grad_acc_steps == grad_acc_steps - 1:
                self._num_iterations += 1

            if self._num_iterations == self.grpo_config.num_iterations:
                self._num_iterations = 0
            return self._data_buffer[idx]
        else:
            return self._generate_and_compute_advantage(training_input)

    def _post_process_train_step(self, aux: Dict[str, Any]) -> None:
        """Post-process training step.

        Args:
            aux: Auxiliary outputs from training step.
        """
        self._metrics_logger.log("kl", aux["kl"], self._mode, self._train_steps)

    def _post_process_eval_step(self, aux: Any) -> None:
        """Post-process training step.

        Args:
            aux: Auxiliary outputs from training step.
        """
        self._metrics_logger.log("kl", aux["kl"], self._mode, self._train_steps)

    def _preprocess_dataset(self, train_ds: Iterable[Any],
        eval_ds: Optional[Iterable[Any]] = None,):
        train_ds = RepeatTrainingInputIter(
            train_ds,
            sample_repeat=self.grpo_config.num_generations,
            batch_repeat=self.grpo_config.num_iterations,
            gradient_accumulation_steps=self.grpo_config.get_with_default(
                "gradient_accumulation_steps", 1
            ),
        )
        eval_ds = RepeatTrainingInputIter(
            eval_ds,
            sample_repeat=self.grpo_config.num_generations,
            batch_repeat=self.grpo_config.num_iterations,
        ) if eval_ds else None

        return train_ds, eval_ds

    def _generate_and_compute_advantage(self, training_input: Dict[str, Any]) -> TrainExample:
        """Generates text completions and computes the advantages for GRPO training.

        Args:
            training_input: A dictionary containing the training input data,
                containing the key 'prompts'.

        Returns:
            A dictionary containing the processed input data, including
            prompt IDs, completion IDs, masks, advantages, and per-token log
            probabilities from the reference and policy models.
        """
        # Generate completions using rollout client
        prompts = training_input["prompts"]
        completions = self.rollout_client.generate(
            prompts=prompts,
            max_tokens=self.config.total_generation_steps,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            top_k=self.config.top_k,
        )

        # Get reference model log probabilities and policy model log probabilities for multiple iterations
        if self.config.beta != 0.0 or self.grpo_config.num_generations > 1:
            ref_per_token_logps, old_per_token_logps, completion_mask, prompt_completion_mask = self.train_client.get_per_token_logps(
                prompts=prompts,
                completions=completions,
            )
        else:
            ref_per_token_logps = None

        # Compute rewards
        rewards = self._compute_rewards(
            prompts=prompts,
            completions=completions,
            **{k: v for k, v in training_input.items() if k != "prompts"},
        )

        # Compute advantages
        advantages = compute_advantages(rewards, self.config.num_generations)

        # Log completion metrics
        steps = self._get_metric_logging_steps()
        self._log_completion_metrics(completion_mask, steps)

        return TrainExample(
            prompt_ids=prompts,
            prompt_mask=prompt_completion_mask[:, : len(prompts[0])],
            completion_ids=completions,
            completion_mask=prompt_completion_mask[:, len(prompts[0]) :],
            ref_per_token_logps=ref_per_token_logps,
            advantages=advantages,
            old_per_token_logps=old_per_token_logps,
        )

    def _compute_rewards(
        self, prompts: List[str], completions: List[str], **kwargs
    ) -> jnp.ndarray:
        """Computes the rewards for completions using the provided reward functions.

        Args:
            prompts: A list of input prompts.
            completions: A list of generated text completions.
            **kwargs: Additional keyword arguments passed to the reward functions.

        Returns:
            A JAX array of scalar rewards for each prompt-completion pair.
        """
        rewards = jnp.zeros((len(prompts), len(self.reward_fns)))
        steps = self._train_steps
        for i, reward_fn in enumerate(self.reward_fns):
            r = reward_fn(prompts=prompts, completions=completions, **kwargs)
            r = jnp.array(r)
            rewards = rewards.at[:, i].set(r)

            self._metrics_logger.log_metric(
                f"rewards/{reward_fn.__name__}",
                r.mean(),
                "train",
            )

        rewards = jnp.nansum(rewards, axis=1)

        self._metrics_logger.log(
            "rewards/overall",
            rewards.mean(),
            self._mode,
            steps,
        )

        return rewards

    def _get_policy_logps(
        self, prompts: List[str], completions: List[str]
    ) -> jnp.ndarray:
        """Get per-token log probabilities from policy model.

        Args:
            prompts: Input prompts.
            completions: Generated completions.

        Returns:
            Per-token log probabilities.
        """
        # TODO: Implement policy model log probability computation
        raise NotImplementedError("Policy model log probability computation not implemented yet")

    def _log_completion_metrics(self, completions: List[str], steps: int) -> None:
        """Log metrics about generated completions.

        Args:
            completions: Generated completions.
        """
        agg_completion_mask = completions.sum(axis=-1)
        self._metrics_logger.log_metric(
            "completions/mean_length",
            np.mean(agg_completion_mask),
            self._mode,
            steps,
        )
        self._metrics_logger.log_metric(
            "completions/max_length",
            np.max(agg_completion_mask),
            self._mode,
            steps,
        )
        self._metrics_logger.log_metric(
            "completions/min_length",
            np.min(agg_completion_mask),
            self._mode,
            steps,
        )

    def _log_metrics(self, loss: float, step: int) -> None:
        """Log training metrics.

        Args:
            loss: Training loss.
            step: Current step.
        """
        self._metrics_logger.log_metric("loss", loss, "train")

    def _may_update_pbar(self, metrics: Dict[str, float], increment_steps: bool = True) -> None:
        """Update progress bar if it exists.

        Args:
            metrics: Current metrics.
            increment_steps: Whether to increment step count.
        """
        if hasattr(self, '_pbar'):
            self._pbar.update(metrics, increment_steps)

    def _run_eval(
        self, eval_ds: Iterable[Any]
    ) -> None:
        """Runs evaluation loop."""
        with self._switch_mode(self._metrics_logger.Mode.EVAL):
            eval_loss, local_eval_steps = 0, 0
            for eval_example in eval_ds:
                eval_example = self._prepare_inputs(eval_example)
                # TODO: moved to server
                # eval_example = self._shard_input(eval_example)
                # TODO: Implement on server
                loss, aux = self.train_client.eval(eval_example) 
                self._eval_steps += 1
                self._post_process_eval_step(aux)
                eval_loss += loss
                local_eval_steps += 1
            self._log_metrics(eval_loss / local_eval_steps, self._train_steps)
            self._may_update_pbar(self._tqdm_eval_metrics)

            logging.info(
                "Train step %d eval loss: %f - eval perplexity: %f",
                self._train_steps,
                self._metrics_logger.get_metric("loss", "eval"),
                self._metrics_logger.get_metric("perplexity", "eval"),
            )

    def learn(
        self,
        train_ds: Iterable[Any],
        eval_ds: Optional[Iterable[Any]] = None,
    ) -> None:
        """Learning loop.

        Args:
            train_ds: Training dataset.
            eval_ds: Optional evaluation dataset.
        """
        logging.info("Starting GRPO learning")
        train_ds, eval_ds = self._preprocess_dataset(train_ds, eval_ds)

        # TODO: To Implement
        self._train_steps = self.train_client.maybe_restore() 

        if self.config.max_steps is not None:
            self._pbar = ProgressBar(
                metrics_logger=self._metrics_logger,
                initial_steps=self._train_steps,
                max_steps=self.config.max_steps,
            )

        with time_measure("Learning loop"):
            for index, train_example in enumerate(train_ds):
                # Skip already trained examples
                if index < self._train_steps:
                    continue

                # Run evaluation if needed
                if (
                    eval_ds
                    and self._train_steps % self.config.eval_every_n_steps == 0
                ):
                    self._run_eval(eval_ds)

                # Stop if max steps reached
                if (
                    self.config.max_steps is not None
                    and self._train_steps >= self.config.max_steps
                ):
                    break


                # Prepare and shard inputs
                train_example = self._prepare_inputs(train_example)
                # TODO: To Implement shard_input on server
                # train_example = self._shard_input(train_example) 

                # Training step
                self._throttler.wait_for_next()
                train_loss, aux = self.train_client.train_step(train_example)
                self._throttler.add_computation(train_loss)
                self._train_steps += 1

                # Post-process and log metrics
                self._post_process_train_step(aux)
                self._log_metrics(train_loss, self._train_steps)
                self._may_update_pbar(self._tqdm_train_metrics, increment_steps=True)

                logging.info(
                    "Learning step %d training loss: %f",
                    self._train_steps,
                    self._metrics_logger.get_metric("loss", "train"),
                )

                self.train_client.sync_weights()

        self._throttler.wait_for_all()
        self.close()

    def close(self) -> None:
        """Clean up resources."""
        if hasattr(self, '_pbar'):
            self._pbar.close()
        if hasattr(self, 'train_client'):
            del self.train_client
        if hasattr(self, 'rollout_client'):
            del self.rollout_client 