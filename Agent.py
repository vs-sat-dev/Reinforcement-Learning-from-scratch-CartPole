import numpy as np
import torch


class Agent:
    def __init__(self, env, exp_buffer, start_step, decay_rate, device='cpu'):
        self.env = env
        self.exp_buffer = exp_buffer
        self.obs = None
        self.start_step = start_step
        self.decay_rate = decay_rate
        self.steps = 0
        self.reset_steps = 0
        self.device = device
        self.solved = False
        self.episodes_list = []
        self.episodes_count = 0

    def reset(self):
        self.obs = self.env.reset()

    def step(self, net):
        self.steps += 1
        if self.steps < self.start_step:
            action = self.env.action_space.sample()
        else:
            action = self.action_scheduler(net)
        old_obs = self.obs
        self.obs, reward, is_done, _ = self.env.step(action)
        self.exp_buffer.append(old_obs, action, reward, self.obs, is_done)

        if is_done:
            self.obs = self.env.reset()
            self.episodes_list.append(self.steps - self.reset_steps)
            self.episodes_count += 1
            if len(self.episodes_list) > 100:
                del self.episodes_list[0]
                print(f'Mean steps for last 100 episodes: {np.mean(self.episodes_list)} episodes:{self.episodes_count}')
            if np.mean(self.episodes_list) >= 195:
                self.solved = True
                print(f'Solved in {self.episodes_count} episodes and {self.steps} steps')
            self.reset_steps = self.steps

    @torch.no_grad()
    def action_scheduler(self, net):
        choice_threshold = np.min([self.steps / (self.decay_rate - self.start_step), 1])
        if choice_threshold > np.random.rand():
            action = int(torch.argmax(net(torch.tensor(self.obs).to(self.device))))
        else:
            action = self.env.action_space.sample()

        return action
