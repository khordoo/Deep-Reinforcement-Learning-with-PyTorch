# Deep Reinforcement Learning with PyTorch

In this repository we will implement various deep reinforcement learning algorithms. 
The main objective is to provide a clean implementation of the deep learning algorithms for deep learning enthusiast who want to have a deeper understanding of RL algorithms 

## **Implemented Algorithms**
1. Cross entropy
2. Deep Q-Learning (DQN)
   - Basic DQN
   - Double DQN
   - Dueling DQN
3. Policy Gradient


### Double DQN
The motivation for double DQN comes from DeepMind.They demonstrated that the basic DQN tend to overestimate the
predicted Q values. This seems to stem from the Max operation in Bellman update equation. The proposed to use the net work to 
predict the best action for the next state and then use the q value associated with this action in bellman equation.
You can read the full details reference paper[2]

In this approach, instead of simply selecting the action associated with the maxQ value. We use the main network to predict the actions for the next state and then 
we choose the Q value associated with this action for our Ballman update approximation.

In my experiments, it took almost the same about of time to train the networks to achieve a certain 
level of reward using both basic and double DQN.However, I used this method on simple environment and it might make huge difference more complicated environments.

**Rewards**

Next figure compares the rewards trend in double DQN (blue) versus to DQN (red)
![image](https://user-images.githubusercontent.com/32692718/78838669-e8cbd200-79b3-11ea-9144-83f91b26d961.png)

### Dueling DQN
The idea of Dueling DQN is to split the Q into tow components Advantage and the value to improve the training stability and results in faster convergence of the network.
The implementation is rather minimal and straightforward. We just need to slightly modify our DQN network to return two values V and A and use these values in our loss calculation.


**Architecture**

Next figure compares the architecture of the simple DQN(top) with that of Duel DQN(bottom)
![image](https://user-images.githubusercontent.com/32692718/78841423-7e6a6000-79ba-11ea-8a29-4d762ed4f7c5.png)
Architecture of the Dueling DQN (source[3])

The main idea is that the average of the advantages that the network predicts for all the actions should be zero. Essentialy at any given state, we can not expect all the actions to be good, 
some of them should be good some of them should be bad and overall average advantage of all actions should be zero.
By imposing this constraint on the network , we force it ot have a better estimation for the V values and in return Q values. 
Next figure shows that the Dueling DQN required less training and converges faster to the desired target value.

**Rewards**
Dueling DQN (orange) vs Base DQN (blue)
![image](https://user-images.githubusercontent.com/32692718/78846761-16237a80-79ca-11ea-88b7-db40a5ec567f.png)

We can see that the effect of Dueling DQN is somehow similar to the effect of the Double DQN.

### N-Step DQN
This improvement comes from a relatively old paper published way back in 1988. The main idea here is to speed up the convergence by providing more terms(n-step)
for Ballman approximation. This generally results in speeding up the convergence by selecting a relativly modest step size n typically 2 or 3. Howver, selecing a very large number 
lets say 100 will totally degrade the performance and the model might not converge at all.

**Rewards**
N-step DQN (gray) vs Base DQN (blue)
![image](https://user-images.githubusercontent.com/32692718/78937769-458bc300-7a6e-11ea-9f38-2136bf4eea79.png)

As you can see the n-step DQN with n=3 is almost two times faster than the base DQN.

## Papers
1. [Playing Atari with Deep Reinforcement Learning](https://arxiv.org/pdf/1312.5602v1.pdf) ,Volodymyr Mnih et. al, 2015
2. [Deep Reinforcement Learning with Double Q-Learning](https://arxiv.org/abs/1509.06461), van Hasselt, Guez, and Silver, 2015 
3. [Dueling Network Architectures for Deep Reinforcement Learning](https://arxiv.org/abs/1511.06581) Wang et al., 2015
4. [Learning to Predict by the Methods of Temporal Differences](http://incompleteideas.net/papers/sutton-88-with-erratum.pdf) Sutton, 1988]