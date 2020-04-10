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
EPSILON_DECAY_FINAL_STEP = 1000
REPLAY_BUFFER_CAPACITY = 100000
SYNC_NETWORKS_EVERY_STEP = 1000
DISCOUNT_FACTOR = 0.99
LEARNING_RATE = 0.001
DESIRED_TARGET_REWARD = 195
N_DISCOUNT_STEPS = 1
PRIORITY_ALPHA = 0.6
PRIORITY_BETA_START = 0.4
PRIORITY_BETA_GROW_LAST_STEP = 10000
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


class DQN(nn.Module):

    def __init__(self, observation_size, hidden_size, action_size):
        super(DQN, self).__init__()
        self.seq = nn.Sequential(
            nn.Linear(observation_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_size)
        )

    def forward(self, x):
        return self.seq(x)


class EpsilonGreedy:
    def __init__(self, start_value, final_value, final_step, decay_mode='default'):
        self.start_value = start_value
        self.final_value = final_value
        self.final_step = final_step
        self.decay_mode = decay_mode.strip().lower()
        self.exponential_decay_rate = np.log(final_value / start_value) / final_step

    def decay(self, step):
        if self.decay_mode == 'exponential':
            epsilon = self.start_value * np.exp(self.exponential_decay_rate * step)
        else:
            epsilon = 1 + step * (self.final_value - self.start_value) / self.final_step

        return max(self.final_value, epsilon)


class EpisodeSteps:
    Step = collections.namedtuple('Step', field_names=['state', 'action', 'reward', 'done', 'next_state'])

    def __init__(self, discount_steps=4):
        self.discount_steps = discount_steps
        self.state = None
        self.action = None
        self.reward = None
        self.done = None
        self.next_state = None
        self._steps = []

    def append(self, state, action, reward, done, next_state):
        self._steps.append(self.Step(state=state, action=action, reward=reward, done=done, next_state=next_state))

    def roll_out(self, discount_factor):
        """Perform n-step roll outs
        """
        first_step_total_discounted_rewards = self._discounted_rewards(discount_factor)
        self._collapse_n_steps(first_step_total_discounted_rewards)

    def _discounted_rewards(self, discount_factor):
        total_discounted_reward_first_state = 0
        for step in reversed(self._steps):
            total_discounted_reward_first_state = step.reward + total_discounted_reward_first_state * discount_factor
        return total_discounted_reward_first_state

    def _collapse_n_steps(self, total_discounted_reward):
        """Collapses n-step into a single step and assigns the the calculated
           total discounted reward to the first state.
           We add the next_state observed in the last step as a next_state.
        """
        self.state = self._steps[0].state
        self.action = self._steps[0].action
        self.reward = total_discounted_reward
        self.done = self._steps[-1].done
        self.next_state = self._steps[-1].next_state

    def completed(self):
        """If episodes ends before reaching n-steps, (i.e done=True)
           we consider the n-steps to be completed to avoid appending
           an irrelevant next_state from the new episode to our last step.
        """
        if self._steps[-1].done:
            return True
        return len(self._steps) == self.discount_steps

    def __len__(self):
        return len(self._steps)


class PriorityReplayBuffer:
    def __init__(self, capacity, priority_alpha=0.6, priority_beta_start=0.4, priority_beta_grow_last_step=10000,
                 device='cpu'):
        self.capacity = capacity
        self.position = 0
        self.priority_alpha = priority_alpha
        self.priority_beta_start = priority_beta_start
        self.priority_beta_grow_last_step = priority_beta_grow_last_step
        self.device = device
        self.buffer = collections.deque(maxlen=capacity)
        self.priorities = np.zeros(self.capacity)

    def append(self, episode_step):
        max_priority = self.get_max_priority()
        if len(self.buffer) < self.capacity:
            self.buffer.append(episode_step)
        else:
            self.buffer[self.position] = episode_step

        self.priorities[self.position] = max_priority
        self.position = (self.position + 1) % self.capacity

    def sample(self, sample_size):
        probabilities = self.get_probabilities()
        indexes = np.random.choice(len(self.buffer), sample_size, p=probabilities)
        samples = [self.buffer[idx] for idx in indexes]

        beta = self.get_updated_beta()
        samples_probabilities = probabilities.take(indexes)
        sample_weights = (self._items_count() * samples_probabilities) ** (-beta)
        return self._vectorize(samples, indexes, sample_weights)

    def get_max_priority(self):
        # Initial: If priorities is empty return 1
        return np.array(self.priorities).max(initial=1)

    def update_priorities(self, indexes, priorities):
        for idx, priority in zip(indexes, priorities):
            self.priorities[idx] = priority

    def get_probabilities(self):
        priorities = self.priorities[:self._items_count()]
        priorities = priorities ** self.priority_alpha
        return priorities / priorities.sum()

    def get_updated_beta(self):
        step = len(self.buffer)
        beta = self.priority_beta_start + (1 - self.priority_beta_start) * (step / self.priority_beta_grow_last_step)
        return min(1, beta)

    def _items_count(self):
        if len(self.buffer) < self.capacity:
            return self.position
        return self.capacity

    def _vectorize(self, samples, indexes, weights):
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
        weights = torch.FloatTensor(np.array(weights, copy=False)).to(self.device)
        return states, actions, rewards, dones, next_states, indexes, weights

    def __len__(self):
        return len(self.buffer)


class Session:
    def __init__(self, env, buffer, net, target_net, epsilon_tracker, device, batch_size, sync_every, discount_factor,
                 learning_rate, discount_steps):
        self.env = env
        self.buffer = buffer
        self.net = net
        self.target_net = target_net
        self.epsilon_greedy = epsilon_tracker
        self.device = device
        self.batch_size = batch_size
        self.sync_steps = sync_every
        self.discount_steps = discount_steps
        self.discount_factor = discount_factor
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=learning_rate)
        self.writer = SummaryWriter(comment='-dqn-n-step-' + datetime.now().isoformat(timespec='seconds'))
        self._reset()
        self.episode_steps = EpisodeSteps(self.discount_steps)

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

            if len(self.buffer) < self.batch_size:
                print('\rFilling up the replay buffer...', end='')
                continue

            states, actions, rewards, dones, next_states, sample_indexes, sample_weights = self.buffer.sample(
                self.batch_size)
            loss, sample_priorities = self._calculate_loss(states, actions, next_states, dones, rewards, sample_weights)
            self.buffer.update_priorities(sample_indexes, sample_priorities)
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
        next_state, reward, done, _ = self.env.step(action)
        self.total_episode_reward += reward

        self.episode_steps.append(self.state, action, reward, done, next_state)
        if self.episode_steps.completed():
            self.episode_steps.roll_out(discount_factor=self.discount_factor)
            self.buffer.append(self.episode_steps)
            self.episode_steps = EpisodeSteps(self.discount_steps)

        if done:
            episode_reward = self.total_episode_reward
            self._reset()
        else:
            self.state = next_state
        return episode_reward

    def _calculate_loss(self, states, actions, next_states, dones, rewards, sample_weights):
        state_q_all = self.net(states)
        state_q_taken_action = state_q_all.gather(1, actions.unsqueeze(-1)).squeeze(-1)

        with torch.no_grad():
            next_state_q_all = self.target_net(next_states)
            next_state_q_max = torch.max(next_state_q_all, dim=1)[0]
            next_state_q_max[dones] = 0
            state_q_expected = rewards + self.discount_factor * next_state_q_max
            state_q_expected = state_q_expected.detach()
        # PyTorch doesn't support weights for  MSELoss class
        loss = (state_q_expected - state_q_taken_action) ** 2
        weighted_loss = sample_weights * loss
        return weighted_loss.mean(), (weighted_loss + 1e-6).data.cpu().numpy()

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
    buffer = PriorityReplayBuffer(capacity=REPLAY_BUFFER_CAPACITY, priority_alpha=PRIORITY_ALPHA,
                                  priority_beta_start=PRIORITY_BETA_START,
                                  priority_beta_grow_last_step=PRIORITY_BETA_GROW_LAST_STEP, device=DEVICE)
    net = DQN(env.observation_space.shape[0], NETWORK_HIDDEN_SIZE, env.action_space.n).to(DEVICE)
    target_net = DQN(env.observation_space.shape[0], NETWORK_HIDDEN_SIZE, env.action_space.n).to(DEVICE)
    epsilon_tracker = EpsilonGreedy(start_value=EPSILON_INITIAL, final_value=EPSILON_FINAL,
                                    final_step=EPSILON_DECAY_FINAL_STEP, decay_mode='default')
    session = Session(env=env, buffer=buffer, net=net, target_net=target_net, epsilon_tracker=epsilon_tracker,
                      device=DEVICE,
                      batch_size=BATCH_SIZE, sync_every=SYNC_NETWORKS_EVERY_STEP, discount_factor=DISCOUNT_FACTOR,
                      learning_rate=LEARNING_RATE, discount_steps=N_DISCOUNT_STEPS)
    session.train(target_reward=DESIRED_TARGET_REWARD)
    # session.demonstrate()
