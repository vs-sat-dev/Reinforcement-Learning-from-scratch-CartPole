# Reinforcement Learning from scratch CartPole solver with Pytorch

This is the solver of the Reinforcement Learning Environment named CartPole.

The environment counted solved if the best 100-episode average reward was 195.27 ± 1.57. I rounded it to 195 and used the last 100-episodes instead best 100-episodes.

Information about the environment can be found here https://gym.openai.com/envs/CartPole-v0/

To make decisions I used a simple Feed-Forward neural network and Q-learning algorithm.

Three consecutive runnings give the following results:

```
Solved in 1337 episodes and 108981 steps

Solved in 1300 episodes and 75544 steps

Solved in 1271 episodes and 110631 steps
```

To improve performance the env.render() from main.py may be commented, it turns off the visualization of the environment but improve the speed of training significantly.

I spent very little time on hyperparameter tuning. I think it may be solved faster if spend some time on tuning.
