from collections import deque
import numpy as np


class ExperienceBuffer:
    def __init__(self, buffer_size):
        self.buffer = deque(maxlen=buffer_size)

    def __len__(self):
        return len(self.buffer)

    def append(self, state, action, reward, next_state, is_done):
        self.buffer.append((state, action, reward, next_state, is_done))

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size)
        states, actions, rewards, next_states, is_dones = zip(*[self.buffer[ind] for ind in indices])
        return np.array(states), np.array(actions), np.array(rewards), \
               np.array(next_states), np.array(is_dones)
