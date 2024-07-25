# Exercise 2.5 (programming) Design and conduct an experiment to demonstrate the difficulties that
# sample-average methods have for nonstationary problems. Use a modified version of the 10-armed
# testbed in which all the q∗ (a) start out equal and then take independent random walks (say by adding
# a normally distributed increment with mean zero and standard deviation 0.01 to all the q∗ (a) on each
# step). Prepare plots like Figure 2.2 for an action-value method using sample averages, incrementally
# computed, and another action-value method using a constant step-size parameter, α = 0.1. Use ε = 0.1
# and longer runs, say of 10,000 steps.

import numpy as np
import matplotlib.pyplot as plt
import random

num_actions = 10
rewards = np.random.uniform(10)
print(rewards)
num_steps = 10000


class Agent:
    def __init__(self):
        pass

    def policy(self, state=None):
        raise NotImplementedError

    def update_estimates(self):
        raise NotImplementedError


class ConstantStepAgent(Agent):
    def __init__(self, alpha):
        self.estimates = np.zeros(10)
        self.alpha = alpha

    def policy(self, state=None):
        return np.argmax(self.estimates)

    def update_estimates(self, action, reward, n):
        self.estimates[action] += 1 / n * (reward - self.estimates[action])


class AveragerAgent(Agent):
    def __init__(self):
        self.estimates = np.zeros(10)

    def policy(self, state=None):
        return np.argmax(self.estimates)

    def update_estimates(self, action, reward, n):
        self.estimates[action] += 1 / n * (reward - self.estimates[action])


class EpsilonAgent(Agent):
    def __init__(self, epsilon):
        self.estimates = np.zeros(10)
        self.epsilon = epsilon

    def policy(self, state=None):
        if np.random.uniform() > self.epsilon:
            return np.argmax(self.estimates)
        else:
            return random.choice([i for i in range(10)])

    def update_estimates(self, action, reward, n):
        self.estimates[action] += 1 / n * (reward - self.estimates[action])


bob = AveragerAgent()
charlie = EpsilonAgent(0.1)

bobs_rewards = []
charlie_rewards = []
rewards_register = []
for i in range(1, num_steps + 1):
    if i % 100 == 0:
        print(i)
    rewards += 0.01 * np.random.randn(10)
    rewards_register.append(rewards.copy())

    action = bob.policy()
    reward = rewards[action]
    bobs_rewards.append(reward)
    bob.update_estimates(action, reward, i)

    action = charlie.policy()
    reward = rewards[action]
    charlie_rewards.append(reward)
    charlie.update_estimates(action, reward, i)
rewards_register = np.array(rewards_register)

fig, (ax1, ax2, ax3) = plt.subplots(nrows=3)
steps = [i for i in range(num_steps)]
ax1.plot(steps, bobs_rewards, "b")
ax2.plot(steps, charlie_rewards, "r")
for i, reward in enumerate(rewards):
    ax3.plot(steps, rewards_register.transpose()[i])

plt.savefig("rewards.png")
