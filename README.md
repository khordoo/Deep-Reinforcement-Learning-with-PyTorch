# Deep Reinforcement Learning with PyTorch

In this repository we will implement various deep reinforcement learning algorithms. 
The main objective is to provide a clean implementation of the deep learning algorithms for deep learning enthusiast who want to have a deeper understanding of RL algorithms 

## **Implemented Algorithms**
1. Cross entropy
2. Deep Q-Learning (DQN)
   - Basic DQN
   - Double DQN
   - Dueling DQN
   - N-step DQN
   - Priority Buffer DQN
   - DQN with prioritized replay buffer
3. Policy Gradient


### Double DQN
The motivation for double DQN comes from DeepMind.They demonstrated that the basic DQN tend to overestimate the
predicted Q values. This seems to stem from the Max operation in Bellman update equation. They proposed to use the network to 
predict the best action for the next_state and then use the q value associated with this action in bellman equation.
You can read the full details in the reference paper[2]

In this approach, instead of simply selecting the action associated with the maxQ value. We use the main network to predict the actions for the next state and then 
we choose the Q value associated with this action for our Ballman update approximation.

In my experiments, it almost took the same amount of time to train both basic and double DQN networks to achieve a certain 
reward level .However, Idid this experiment on a simple environment CartPole, it might make a huge difference in more complicated environments.

**Rewards**

Next figure compares the rewards trend in double DQN (blue) versus to DQN (red)
![image](https://user-images.githubusercontent.com/32692718/78838669-e8cbd200-79b3-11ea-9144-83f91b26d961.png)

### Dueling DQN
The idea of Dueling DQN is to split the Q into two components, Advantage and Value, to improve the training stability and faster convergence of the network.
The implementation is rather minimal and straightforward. We just need to slightly modify our DQN network to return two values V and A and use these values in our loss calculation.


**Architecture**

Next figure compares the architecture of the base DQN(top) with that of Duel DQN(bottom)
![image](https://user-images.githubusercontent.com/32692718/78841423-7e6a6000-79ba-11ea-8a29-4d762ed4f7c5.png)
Architecture of the Dueling DQN (source[3])

The main idea here is that the average of the advantages that the network predicts, for all the actions, should be zero. Essentialy at any given state, we can not expect all the actions to be good, 
some of them should be good some of them should be bad and overall, average advantage of all actions should be zero.
By imposing this constraint on the network , we force it ot have a better estimation for the V values and in which in return results in better estimations for Q values. 
Next figure shows that the Dueling DQN required less training and converges faster to a desired reward value.

**Rewards**

Dueling DQN (orange) vs Base DQN (blue)
![image](https://user-images.githubusercontent.com/32692718/78846761-16237a80-79ca-11ea-88b7-db40a5ec567f.png)

We can see that the effect of Dueling DQN is somehow similar to the effect of the Double DQN.

### N-Step DQN
This improvement comes from a relatively old paper published way back in 1988. The main idea here is to speed up the convergence by providing more terms(n-step)
for Ballman approximation. This generally results in speeding up the convergence by selecting a relativly modest step size n typically 2 or 3. Howver, selecing a very large number, say 100, will totally degrade the performance and the model might not converge at all.

**Rewards**
N-step DQN (gray) vs Base DQN (blue)
![image](https://user-images.githubusercontent.com/32692718/78937769-458bc300-7a6e-11ea-9f38-2136bf4eea79.png)

As you can see the n-step DQN with n=3 is almost two times faster than the base DQN.

### DQN with Prioritized Replay Buffer
This concept was introduced by DeepMind and the idea is that instead of just randomly selecting samples from the replay buffer, we should select samples that are more beneficial for the model training.
So, how do we know that a sample is beneficial(or has higher priority as they call it)? Well, they suggested different approaches in their paper[5] 
One of them is to use the training error as an indicator for selecting a sample. In other words, if a sample has a high error, we want to sample it more frequently to give the network a chance to train on it more often and ultimately reduce the error associated with it and improve the training efficiency.

## Noisy Networks

This approach tries to solve the exploration vs exploitation dilemma in the DQN. Typically DQN allows the agent to do the exploration by selecting random actions
during the training.The amount of random action is controlled by a hyperparameter epsilon and it is generally linearly decreased over time as the training progress.(epsilon-greedy)
The challenge is that we often need to fine tune the epsilon to make the training fast and efficient.
The main benefit of noisy networks is that we do not have to worry about fine tuning another hyperparameter as the network adjusts the required amount of exploration
by agent automatically by itself. Whenever more exploration is required more noise is automatically added to the weights of the fully connected layers of the network and its amount is adjusted by back propagation.This method generally results in faster convergence.
The initialization scheme and initial value for sigma were adopted from the reference paper[7]



**Rewards dynamics**

In my experiments with the CartPole environment. Runs with prioritized replay buffer tend to take less episode to reach a specific target level compared to simple replay buffer.
However, the runs take a bit longer which is expected.

![image](https://user-images.githubusercontent.com/32692718/79031718-f3c26600-7b5d-11ea-8036-c3913fac1edf.png)

Rewards comparision for prioritized replay buffer( blue) vs base (orange)

**Loss dynamics**

![image](https://user-images.githubusercontent.com/32692718/79031743-35531100-7b5e-11ea-9f82-baeeb44e12eb.png)

Rewards comparision for prioritized replay buffer( blue) vs base (orange)


### Rainbow: Combining the above improvements
All the previous DQN extensions that were mentioned so far, were the result of many researchers who has made significant improvements to the DQN algorithm over the years. 
The so called Rainbow paper[5] combined all of these enhancements in a single model and evaluated the performance of the combined model.

In the current implementation I have combined the following methods: 
 - Dueling DQN
 - N-step DQN
 - Prioritized Replay Buffer
 - Noisy Networks
 
 The performance gain is quite considerable. Next figure compares the reward dynamics for this model with the base DQN for the CartPole env.
 
 ![image](https://user-images.githubusercontent.com/32692718/79032733-0e98d880-7b66-11ea-8d3f-840094d5f446.png)
 
 Rainbow(blue) vs baseline vanilla DQN (orange)
 
 The code can still be optimized further by , for instance, improving the sampling in O(log N) time in the priority replay buffer , by using a more efficient data structure like the segment tree. Even without that the
 final performance is still quite good and positive. The time complexity of our buffer is O(N) and this should generally be fine for the learning purposes or if our buffer is relatively small, say 200k.However, you might need to consider a more optimized replay buffer for large projects which typically require larger buffers to hold millions of frames.

## Reference Papers
1. [Playing Atari with Deep Reinforcement Learning](https://arxiv.org/pdf/1312.5602v1.pdf) ,Volodymyr Mnih et. al, 2015
2. [Deep Reinforcement Learning with Double Q-Learning](https://arxiv.org/abs/1509.06461), van Hasselt, Guez, and Silver, 2015 
3. [Dueling Network Architectures for Deep Reinforcement Learning](https://arxiv.org/abs/1511.06581) Wang et al., 2015
4. [Learning to Predict by the Methods of Temporal Differences](http://incompleteideas.net/papers/sutton-88-with-erratum.pdf) Sutton, 1988]
5. [Prioritized Experience Replay](https://arxiv.org/abs/1511.05952) Schaul and others, 2015
6. [Rainbow:Combining Improvements in Deep Reinforcement Learning](https://arxiv.org/abs/1710.02298) Matteo Hessel and others, 2017
7. [Noisy Networks for Exploration](https://arxiv.org/abs/1706.10295) Fortunato and others, 2017
