import torch
import torch.nn as nn
import numpy as np
import collections
import gym
from datetime import datetime
from tensorboardX import SummaryWriter

ENV_NAME = 'CartPole-v0'
NETWORK_HIDDEN_SIZE = 128
BATCH_SIZE = 128
EPSILON_INITIAL = 1
EPSILON_FINAL = 0.02
EPSILON_DECAY_FINAL_STEP = 7000
REPLAY_BUFFER_CAPACITY = 40000
SYNC_NETWORKS_EVERY_STEP = 1000
DISCOUNT_FACTOR = 0.99
LEARNING_RATE = 0.01
DESIRED_TARGET_REWARD = 196
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


class DuelingDQN(nn.Module):

    def __init__(self, observation_size, hidden_size, action_size):
        super(DuelingDQN, self).__init__()
        mid_size = int(hidden_size / 2)
        self.common = nn.Sequential(
            nn.Linear(observation_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, mid_size),
            nn.ReLU(),
        )
        self.advantage = nn.Sequential(
            nn.Linear(mid_size, action_size)
        )
        self.value = nn.Sequential(
            nn.Linear(mid_size, 1)
        )

    def forward(self, x):
        x = self.common(x)
        advantage, value = self.advantage(x), self.value(x)
        return value + (advantage - advantage.mean())


class EpsilonGreedy:
    def __init__(self, start_value, final_value, final_step):
        self.start_value = start_value
        self.final_value = final_value
        self.final_step = final_step

    def decay(self, step):
        epsilon = 1 + step * (self.final_value - self.start_value) / self.final_step
        return max(self.final_value, epsilon)


class EpisodeStep:
    def __init__(self, state, action, reward, done, next_step):
        self.state = state
        self.action = action
        self.reward = reward
        self.done = done
        self.next_state = next_step


class ReplayBuffer:
    def __init__(self, capacity, device='cpu'):
        self.capacity = capacity
        self.device = device
        self.buffer = collections.deque(maxlen=capacity)

    def append(self, episode_step):
        self.buffer.append(episode_step)

    def sample(self, sample_size):
        # Note: replace=False makes random.choice O(n)
        indexes = np.random.choice(self.capacity, sample_size, replace=True)
        samples = [self.buffer[idx] for idx in indexes]
        return self._unpack(samples)

    def _unpack(self, samples):
        states, actions, rewards, dones, next_states = [], [], [], [], []
        for episode_step in samples:
            states.append(episode_step.state)
            actions.append(episode_step.action)
            rewards.append(episode_step.reward)
            dones.append(episode_step.done)
            next_states.append(episode_step.next_state)

        states = torch.FloatTensor(np.array(states, copy=False)).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states, copy=False)).to(self.device)
        actions = torch.LongTensor(np.array(actions, copy=False)).to(self.device)
        rewards = torch.FloatTensor(np.array(rewards, copy=False)).to(self.device)
        dones = torch.BoolTensor(np.array(dones, copy=False)).to(self.device)

        return states, actions, rewards, dones, next_states

    def __len__(self):
        return len(self.buffer)


class Session:
    def __init__(self, env, buffer, net, target_net, epsilon_tracker, device, batch_size, sync_every, discount_factor,
                 learning_rate):
        self.env = env
        self.buffer = buffer
        self.net = net
        self.target_net = target_net
        self.epsilon_greedy = epsilon_tracker
        self.device = device
        self.batch_size = batch_size
        self.sync_steps = sync_every
        self.discount_factor = discount_factor
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=learning_rate)
        self.writer = SummaryWriter(comment='-dqn-dueling-' + datetime.now().isoformat(timespec='seconds'))
        self._reset()

    def _reset(self):
        self.state = self.env.reset()
        self.total_episode_reward = 0

    def train(self, target_reward):
        step = 0
        episode_rewards = []
        while True:
            self.optimizer.zero_grad()

            epsilon = self.epsilon_greedy.decay(step)
            episode_reward = self._play_single_step(epsilon)

            if len(self.buffer) < self.buffer.capacity:
                continue

            states, actions, rewards, dones, next_states = self.buffer.sample(self.batch_size)
            loss = self._calculate_loss(states, actions, next_states, dones, rewards)
            loss.backward()
            self.optimizer.step()
            self._periodic_sync_target_network(step)

            if episode_reward is not None:
                episode_rewards.append(episode_reward)
                mean_reward = np.array(episode_rewards)[-100:].mean()
                self._report_progress(step, loss.item(), episode_rewards, mean_reward, epsilon)
                if mean_reward > target_reward:
                    print('\nEnvironment Solved!')
                    self.writer.close()
                    break

            step += 1

    @torch.no_grad()
    def _play_single_step(self, epsilon):
        episode_reward = None
        state_t = torch.FloatTensor(np.array([self.state], copy=False)).to(self.device)
        q_actions = self.net(state_t)
        action = torch.argmax(q_actions, dim=1).item()
        if np.random.random() < epsilon:
            action = np.random.choice(self.env.action_space.n)
        new_state, reward, done, _ = self.env.step(action)
        self.total_episode_reward += reward
        self.buffer.append(EpisodeStep(self.state, action, reward, done, new_state))
        if done:
            episode_reward = self.total_episode_reward
            self._reset()
        else:
            self.state = new_state
        return episode_reward

    def _calculate_loss(self, states, actions, next_states, dones, rewards):
        state_q_all = self.net(states)
        state_q_taken_action = state_q_all.gather(1, actions.unsqueeze(-1)).squeeze(-1)

        with torch.no_grad():
            next_state_q_all = self.target_net(next_states)
            next_state_q_max = torch.max(next_state_q_all, dim=1)[0]
            next_state_q_max[dones] = 0
            state_q_expected = rewards + self.discount_factor * next_state_q_max
            state_q_expected = state_q_expected.detach()
        return nn.functional.mse_loss(state_q_expected, state_q_taken_action)

    def _periodic_sync_target_network(self, step):
        if step % self.sync_steps:
            self.target_net.load_state_dict(self.net.state_dict())

    def _report_progress(self, step, loss, episode_rewards, mean_reward, epsilon):
        self.writer.add_scalar('Reward', mean_reward, step)
        self.writer.add_scalar('loss', loss, step)
        self.writer.add_scalar('epsilon', epsilon, step)
        print(f'\rsteps:{step} , episodes:{len(episode_rewards)}, loss: {loss:.6f} , '
              f'eps: {epsilon:.2f}, reward: {mean_reward:.2f}', end='')

    def demonstrate(self, net_state_file_path=None):
        """Demonstrate the performance of the trained net in a video"""
        env = gym.wrappers.Monitor(self.env, 'videos', video_callable=lambda episode_id: True, force=True)
        if net_state_file_path:
            state_dict = torch.load(net_state_file_path, map_location=lambda stg, _: stg)
            self.net.load_state_dict(state_dict)
        state = env.reset()
        total_reward = 0
        while True:
            env.render()
            action = self.net(torch.FloatTensor([state])).max(dim=1)[1]
            new_state, reward, done, _ = env.step(action.item())
            total_reward += reward
            if done:
                break
            state = new_state
        print("Total reward: %.2f" % total_reward)


if __name__ == '__main__':
    env = gym.make(ENV_NAME)
    buffer = ReplayBuffer(capacity=REPLAY_BUFFER_CAPACITY, device=DEVICE)
    net = DuelingDQN(env.observation_space.shape[0], NETWORK_HIDDEN_SIZE, env.action_space.n).to(DEVICE)
    target_net = DuelingDQN(env.observation_space.shape[0], NETWORK_HIDDEN_SIZE, env.action_space.n).to(DEVICE)
    epsilon_tracker = EpsilonGreedy(start_value=EPSILON_INITIAL, final_value=EPSILON_FINAL,
                                    final_step=EPSILON_DECAY_FINAL_STEP)
    session = Session(env=env, buffer=buffer, net=net, target_net=target_net, epsilon_tracker=epsilon_tracker,
                      device=DEVICE,
                      batch_size=BATCH_SIZE, sync_every=SYNC_NETWORKS_EVERY_STEP, discount_factor=DISCOUNT_FACTOR,
                      learning_rate=LEARNING_RATE)
    session.train(target_reward=DESIRED_TARGET_REWARD)
    session.demonstrate()
