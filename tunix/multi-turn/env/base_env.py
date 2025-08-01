from abc import ABC, abstractmethod
from typing import Tuple

class BaseEnv(ABC):
    @abstractmethod
    def reset(self) -> str:
        """observation"""
        pass

    @abstractmethod
    async def step(self, action: str) -> Tuple[str, float, bool, dict]:
        """
        return (next_obs, reward, done, info)
        """
        pass
