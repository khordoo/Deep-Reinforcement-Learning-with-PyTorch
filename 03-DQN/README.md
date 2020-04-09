# Deep Q-learning (DQN) models.
In this repository we will implement the DQN models along with the latest extensions and improvements 
that have been added to them through the research community. 
The main objective is to provide an step by step and easy to follow Deep Q Learning tutorial with clean and readable code.

The following models have been implemented: 
1. Base Deep Q-learning (DQN)
2. Deep Reinforcement Learning with Double Q-learning 
3. Dueling Network Architectures for Deep Reinforcement Learning 
4. N-step Deep Q-Learning

## Implementation details
### Epsilon Greedy
The epsilon greedy is implement in its own class and supports two decay mode.
- Linear
- Exponential. 
The default is exponential and can bet set through the class constructor argument. Next figure shows the decay trend for each of the mentioned modes.


### Notebooks
All the models are also provided in Jupyter notebooks to for quick run on Google colab GPU accelerated environment.


### Examples

Example folder provides the solution for some of the classic Reinforcement environments.
The provided solutions are as follows:
