import random
from collections import deque
import numpy as np

class ReplayBuffer:
    """A fixed-size buffer to store experience tuples."""
    def __init__(self, capacity: int):
        self.memory = deque([], maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        state = np.array(state)
        next_state = np.array(next_state)
        self.memory.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> tuple:
        experiences = random.sample(self.memory, k=batch_size)
        states, actions, rewards, next_states, dones = zip(*experiences)
        return (np.array(states), np.array(actions), np.array(rewards),
                np.array(next_states), np.array(dones))

    def __len__(self) -> int:
        return len(self.memory)