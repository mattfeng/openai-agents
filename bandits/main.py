#!/usr/bin/env python

import os
import sys
sys.path.append("{}/torchutils/".format(os.environ["HOME"]))
from torchutils.bootstrap import bootstrap

import torch as T
import numpy as np

import gym
import openai_envs

import matplotlib.pyplot as plt

NUM_STEPS = 1000
EPISODES = 2000
EPS = 0.1

def epsilon_greedy(q, eps):
    # generate policy
    if np.random.rand(1) < eps:
        return np.random.choice(len(q)), True
    
    greedy_action = np.argmax(q)
    return greedy_action, False

def train(M):
    M.env.reset()
    M.Q = [0 for _ in range(M.env.action_space.n)]
    M.N = [0 for _ in range(M.env.action_space.n)]

    rewards = []

    for step in range(NUM_STEPS):
        # sample action from policy using Q
        action, was_random = epsilon_greedy(M.Q, M.eps)

        _, reward, _, info = M.env.step(action)
        M.log("step={}, action={:3d}, random={:2d}, reward={:6.2f}".format(
            step, action, was_random, reward), stdout=False)
        
        # update Q
        M.N[action] += 1
        M.Q[action] += (1 / M.N[action]) * (reward - M.Q[action])

        rewards.append(reward)
    
    return rewards


@bootstrap.main
def main(**kwargs):
    M = kwargs["M"]
    M.env = gym.make("Bandit-v0")
    M.eps = EPS
    M.log.init()

    M.env.render()

    rewards = []

    for ep in range(EPISODES):
        rewards.append(train(M))
    
    plt.plot(np.array(rewards).mean(axis=0))
    plt.xlabel("steps")
    plt.ylabel("average")
    plt.show()

if __name__ == "__main__":
    main()