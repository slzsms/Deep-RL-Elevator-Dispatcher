
import random
from collections import deque


class ReplayBuffer:
    def __init__(self, capacity):
        """Initialize the replay buffer."""

        self.buffer = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        """Store one experience."""

        experience = (
            state,
            action,
            reward,
            next_state,
            done
        )

        self.buffer.append(experience)

    def sample(self, batch_size):
        """Randomly sample a batch of experiences."""

        experiences = random.sample(self.buffer, batch_size)

        states, actions, rewards, next_states, dones = zip(
            *experiences
        )

        return (
            states,
            actions,
            rewards,
            next_states,
            dones
        )

    def __len__(self):
        """Return the number of experiences currently stored."""

        return len(self.buffer)
