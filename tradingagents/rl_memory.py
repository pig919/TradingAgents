import collections
import random
from typing import List, Tuple

class RLMemory:
    """A memory buffer for reinforcement learning."""

    def __init__(self, max_size: int):
        """
        Initialize the RLMemory.

        Args:
            max_size: The maximum size of the memory buffer.
        """
        self.buffer = collections.deque(maxlen=max_size)

    def add(self, state, action, reward, next_state, done):
        """
        Add a new experience to the memory buffer.

        Args:
            state: The current state.
            action: The action taken.
            reward: The reward received.
            next_state: The next state.
            done: Whether the episode is finished.
        """
        experience = (state, action, reward, next_state, done)
        self.buffer.append(experience)

    def sample(self, batch_size: int) -> List[Tuple]:
        """
        Sample a batch of experiences from the memory buffer.
        Args:
            batch_size: The number of experiences to sample.
        Returns:
            A list of sampled experiences.
        """
        return random.sample(self.buffer, batch_size)
