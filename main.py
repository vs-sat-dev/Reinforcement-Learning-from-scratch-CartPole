import gym
import torch
import torch.optim as optim

from SimpleDQN import SimpleDQN
from ExperienceBuffer import ExperienceBuffer
from Agent import Agent


if __name__ == '__main__':
    env = gym.make('CartPole-v0')
    obs = env.reset()

    TRAIN_START = 1000
    BATCH_SIZE = 16
    LEARNING_RATE = 0.0001
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    GAMMA = 0.9
    SYNC_STEPS = 100

    exp_buffer = ExperienceBuffer(buffer_size=TRAIN_START)
    agent = Agent(env, exp_buffer, TRAIN_START, 10000, device=DEVICE)
    net = SimpleDQN(*env.observation_space.shape, 128, env.action_space.n).to(DEVICE)
    target_net = SimpleDQN(*env.observation_space.shape, 128, env.action_space.n).to(DEVICE)
    optimizer = optim.Adam(params=net.parameters(), lr=LEARNING_RATE)
    criterion = torch.nn.MSELoss()

    for i in range(1000000):
        agent.step(net)
        env.render()

        if i % SYNC_STEPS == 0:
            target_net.load_state_dict(net.state_dict())

        if i >= TRAIN_START:
            states, actions, rewards, next_states, is_dones = exp_buffer.sample(BATCH_SIZE)
            states = torch.tensor(states).to(DEVICE)
            actions = torch.tensor(actions).to(DEVICE)
            rewards = torch.tensor(rewards).to(DEVICE)
            next_states = torch.tensor(next_states).to(DEVICE)

            action_values = net(states).gather(dim=1, index=actions.unsqueeze(-1)).squeeze(-1)
            next_action_values = torch.max(target_net(next_states), dim=1)[0]
            next_action_values[is_dones] = 0.0
            next_action_values = next_action_values.detach()

            expected_action_values = next_action_values * GAMMA + rewards
            loss = criterion(action_values.float(), expected_action_values.float())
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        if agent.solved:
            break
