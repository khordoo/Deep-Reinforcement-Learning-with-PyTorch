import time
import gym
import torch
import torch.nn as nn
import torch.nn.functional as nnf
import numpy as np
from tensorboardX import SummaryWriter

ENV_NAME = 'CartPole-v1'
BATCH_SIZE = 200
NETWORK_HIDDEN_SIZE = 24
TARGET_REWARD_LEVEL = 200
FPS = 25


class Net(nn.Module):
    """Simple sequential network that will act as our policy
    """

    def __init__(self, observation_size, hidden_size, action_size):
        super(Net, self).__init__()
        self.seq = nn.Sequential(
            nn.Linear(observation_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_size)
        )

    def forward(self, x):
        return self.seq(x)


class Episode:
    """A class that stores the sequence of the steps that we took in the environment
    """

    def __init__(self):
        self.total_reward = 0
        self.states = []
        self.actions = []
        self.rewards = []

    def append(self, state, action, reward):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.total_reward += reward


class Session:
    """Main class that plays the game for some episodes and learns how to play using the cross entropy method.
       Some GPU optimizations were not considered in the implementation to make the code more clear.
    """

    def __init__(self, env, net, batch_size, target_reward_level):
        self.env = env
        self.net = net
        self.batch_size = batch_size
        self.target_reward_level = target_reward_level
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=0.01)
        self.writer = SummaryWriter()

    def train(self):
        step = 0
        while True:
            self.optimizer.zero_grad()
            batch = self.play()
            mean_reward = np.array([episode.total_reward for episode in batch]).mean()
            episode_states, episode_actions = [], []
            for episode in batch:
                if episode.total_reward > mean_reward:
                    episode_states.extend(episode.states)
                    episode_actions.extend(episode.actions)

            predicted_action_scores = self.net(torch.FloatTensor(episode_states))
            episode_actions = torch.LongTensor(episode_actions)
            loss = nn.CrossEntropyLoss()(predicted_action_scores, episode_actions)
            loss.backward()
            self.optimizer.step()

            if mean_reward > self.target_reward_level:
                print('\nSolved!')
                torch.save(self.net.state_dict(), self.env.spec.id + '.dat')
                self.writer.close()
                break
            step += 1
            print(f'{step} : loss: {loss.item()}, reward:{mean_reward}')
            self.writer.add_scalar('Mean reward', mean_reward, step)
            self.writer.add_scalar('LOss', loss.item(), step)

    def play(self):
        """Play some episodes to get the training data"""
        batch = []
        episode = Episode()
        state = self.env.reset()
        number_of_actions = self.env.action_space.n

        while True:
            actions_score = self.net(torch.FloatTensor(state))
            actions_prob = nnf.softmax(actions_score, dim=0)

            action = np.random.choice(number_of_actions, size=1, p=actions_prob.data.numpy())[0]

            new_state, reward, done, _ = self.env.step(action)
            episode.append(state, action, reward)

            if done:
                if len(batch) == self.batch_size:
                    return batch
                batch.append(episode)
                new_state = self.env.reset()
                episode = Episode()

            state = new_state

    def demonstrate(self):
        """Demonstrate the performance of the trained net in a video"""
        env = gym.wrappers.Monitor(self.env, 'videos', video_callable=lambda episode_id: True, force=True)
        state = env.reset()
        total_reward = 0
        while True:
            start_ts = time.time()
            env.render()
            action_scores = self.net(torch.FloatTensor([state]))
            actions_prob = torch.softmax(action_scores, dim=1)
            action = np.random.choice(self.env.action_space.n, p=actions_prob.data.numpy()[0])
            state, reward, done, _ = env.step(action)
            total_reward += reward
            delta = 1 / FPS - (time.time() - start_ts)
            if delta > 0:
                time.sleep(delta)
            if done:
                break
        print("Total reward: %.2f" % total_reward)


if __name__ == '__main__':
    env = gym.make(ENV_NAME)
    net = Net(observation_size=env.observation_space.shape[0],
              hidden_size=NETWORK_HIDDEN_SIZE,
              action_size=env.action_space.n)
    session = Session(env=env, net=net, batch_size=BATCH_SIZE,
                      target_reward_level=TARGET_REWARD_LEVEL)
    session.train()
    session.demonstrate()
