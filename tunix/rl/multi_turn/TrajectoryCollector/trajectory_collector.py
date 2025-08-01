import asyncio
from abc import ABC, abstractmethod
from typing import Any

from tunix.rl.multi_turn.agents.base_agent import BaseAgent, Trajectory
from tunix.rl.multi_turn.environments.base_environment import BaseEnv


class TrajectoryCollector(ABC):
    """
    Abstract base class for collecting trajectories from an agent interacting with an environment.
    This is the core interface for implementing rollout logic in RLHF / multi-turn fine-tuning setups.
    """

    def __init__(self, agent: BaseAgent, env: BaseEnv, max_steps: int = 10, gamma: float = 1.0):
        """
        Initialize the trajectory collector.

        Args:
            agent (BaseAgent): The agent instance.
            env (BaseEnv): The environment instance.
            max_steps (int): Maximum number of steps per trajectory.
            gamma (float): Discount factor for computing returns.
        """
        self.agent = agent
        self.env = env
        self.max_steps = max_steps
        self.gamma = gamma

    @abstractmethod
    async def reset(self) -> None:
        """
        Reset agent and environment states before collecting a new trajectory.
        """
        pass

    @abstractmethod
    async def run_step(self, step_idx: int) -> tuple[Any, float, bool, dict]:
        """
        Execute one step of interaction between the agent and environment.

        Args:
            step_idx (int): The index of the current step.

        Returns:
            observation, reward, done, info
        """
        pass

    @abstractmethod
    async def collect_trajectory(self) -> Trajectory:
        """
        Run a full rollout of agent-environment interaction and return the resulting trajectory.

        Returns:
            Trajectory: A data structure representing the full episode.
        """
        pass

    @abstractmethod
    async def compute_final_reward(self) -> float:
        """
        Optionally compute a final reward at the end of the episode (e.g., from environment metadata).

        Returns:
            float: Final reward score.
        """
        pass

    @abstractmethod
    async def close(self) -> None:
        """
        Clean up environment or agent states, if necessary.
        """
        pass
