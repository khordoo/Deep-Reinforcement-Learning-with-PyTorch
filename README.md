# Deep Reinforcement Learning with PyTorch

In this repository we will implement various deep reinforcement learning algorithms. 
The main objective is to provide a clean implementation of the deep learning algorithms for deep learning enthusiast who want to have a deeper understanding of RL algorithms 

## **Implemented Algorithms**
1. Cross entropy
2. Deep Q-Learning (DQN)
   - Basic DQN
   - Double DQN
3. Policy Gradient


### Double DQN
The motivation for double DQN comes from DeepMind.They demonstrated that the basic DQN tend to overestimate the
predicted Q values. This seems to stem from the Max operation in Bellman update equation. The proposed to use the net work to 
predict the best action for the next state and then use the q value associated with this action in bellman equation.
You can read the full details reference paper[2]

In my experiments, it took almost the same about of time to train the networks to achieve a certain 
level of reward using both basic and double DQN.However, I used this method on simple environment and it might make huge difference more complicated environments.

**Rewards**
Next figure compares the rewards trend in double DQN (blue) versus to DQN (red)
![image](https://user-images.githubusercontent.com/32692718/78838669-e8cbd200-79b3-11ea-9144-83f91b26d961.png)


## Papers
1. [Playing Atari with Deep Reinforcement Learning](https://arxiv.org/pdf/1312.5602v1.pdf) ,Volodymyr Mnih et. al, 2015
2. [Deep Reinforcement Learning with Double Q-Learning](https://arxiv.org/abs/1509.06461), van Hasselt, Guez, and Silver, 2015 