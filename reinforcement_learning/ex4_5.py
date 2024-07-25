# dynamic programing for policy iteration in reinforcement learning exercise

# Jack manages two locations for a nationwide car rental company.
# Each day, some number of customers arrive at each location to rent cars. If Jack has a car available, he
# rents it out and is credited $10 by the national company. If he is out of cars at that location, then the
# business is lost. Cars become available for renting the day after they are returned. To help ensure that
# cars are available where they are needed, Jack can move them between the two locations overnight, at
# a cost of $2 per car moved. We assume that the number of cars requested and returned at each location
# n
# are Poisson random variables, meaning that the probability that the number is n is λn! e−λ , where λ is
# the expected number. Suppose λ is 3 and 4 for rental requests at the first and second locations and
# 3 and 2 for returns. To simplify the problem slightly, we assume that there can be no more than 20
# cars at each location (any additional cars are returned to the nationwide company, and thus disappear
# from the problem) and a maximum of five cars can be moved from one location to the other in one
# night. We take the discount rate to be γ = 0.9 and formulate this as a continuing finite MDP, where
# the time steps are days, the state is the number of cars at each location at the end of the day, and
# the actions are the net numbers of cars moved between the two locations overnight.
import math
import numpy as np
from collections import defaultdict

max_cars = 10


def poisson(x, lambda_):
    return lambda_**x / math.factorial(x) * np.exp(-lambda_)


def enviorment(state, action):
    probs = defaultdict(float)
    for request1 in range(max_cars):
        for return1 in range(max_cars):
            s1_next = min(max_cars, max(0, state[0] - request1 + return1 - action))
            prob_r1 = poisson(request1, 3)
            prob_b1 = poisson(return1, 3)
            for request2 in range(max_cars):
                for return2 in range(max_cars):
                    prob_r2 = poisson(request2, 4)
                    prob_b2 = poisson(return2, 2)
                    s2_next = min(
                        max_cars, max(0, state[1] - request2 + return2 + action)
                    )
                    reward = 10 * (
                        min(state[0], request1) + min(state[1], request2)
                    ) - 2 * abs(action)
                    probs[((s1_next, s2_next), reward)] += (
                        prob_r1 * prob_b1 * prob_r2 * prob_b2
                    )
    return dict(probs)


def posible_actions(state):
    min_action = max(-5, -state[1])
    max_action = min(5, state[0])

    actions = [i for i in range(min_action, max_action + 1)]

    return np.array(actions).reshape(-1)


def policy_evaluation(values, policy, accuracy=5, gamma=0.9):
    print("evaluating")
    while True:
        delta = 0
        for i, state in enumerate(states):
            print(i)
            v = values[state]
            probs = enviorment(state, policy[state])
            sum = 0.0
            for (next_state, reward), prob in probs.items():
                sum += prob * (reward + gamma * values[next_state])
            values[state] = sum
            delta = max(delta, np.abs(v - values[state]))
        print(delta)
        if delta < accuracy:
            break
    return values


def policy_improvement(values, policy, gamma=0.9):
    print("improving")
    policy_stable = True
    for i, state in enumerate(states):
        print(i)
        max_action_val = -np.inf
        old_action = policy[state]
        max_action = old_action
        for action in posible_actions(state):
            probs = enviorment(state, policy[state])
            sum = 0.0
            for (next_state, reward), prob in probs.items():
                sum += prob * (reward + gamma * values[next_state])
            if sum > max_action_val:
                max_action_val = sum
                max_action = action
        policy[state] = max_action
        if old_action != policy[state]:
            print("enters")
            policy_stable = False
    if policy_stable:
        print("stable")
        print(values)
        print(policy)
        return values, policy
    else:
        values = policy_evaluation(values, policy)
        return policy_improvement(values, policy)


states = [(i, j) for j in range(max_cars + 1) for i in range(max_cars + 1)]
values = {state: np.random.random() for state in states}
policy = {state: np.random.choice(posible_actions(state)) for state in states}
values = policy_evaluation(values, policy)
policy_improvement(values, policy)
