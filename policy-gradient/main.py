#!/usr/bin/env python

import gym
import tensorflow as tf
import numpy as np
from agent import VanillaPolicyGradientAgent
from collections import deque

NUM_EPOCHS = 1000
DISCOUNT_FACTOR = 0.99
BATCH_SIZE = 10

def accumulate(rewards, gamma):
    advantages = np.zeros_like(rewards)

    acc = 0
    for ix, r in enumerate(rewards[::-1]):
        acc = gamma * acc + r
        advantages[len(rewards) - ix - 1] = acc

    advantages -= np.mean(advantages)
    advantages /= np.std(advantages)
    
    return advantages


def main():
    env = gym.make("CartPole-v0")
    sess = tf.Session()
    hparams = {
        "learning_rate": 0.03
    }
    agent = VanillaPolicyGradientAgent(env, sess, hparams)
    reward_buffer = deque([], maxlen=100)

    for epoch in range(1, NUM_EPOCHS + 1):
        states, actions = [], []
        advantages = []
        for episode in range(1, BATCH_SIZE + 1):
            # perform rollout
            s = env.reset()
            rewards = []

            done = False
            while not done:
                env.render()
                a = agent.act(s)
                s_, r, done, _ = env.step(a)
                states.append(s)
                actions.append(a)
                rewards.append(r)
                s = s_

            adv = accumulate(rewards, DISCOUNT_FACTOR)
            advantages.extend(adv)
            reward = sum(rewards)
            reward_buffer.append(reward)
            avg_reward = sum(reward_buffer) / len(reward_buffer)
            print(f"\t [episode_{episode}] reward: {reward}, "
                  f"avg_reward: {avg_reward:.1f}")

        states = np.array(states)
        actions = np.array(actions)
        advantages = np.array(advantages)
        
        loss = agent.learn(states, actions, advantages, BATCH_SIZE)
        print(f"[epoch_{epoch}] loss={loss:.4f}")


if __name__ == "__main__":
    main()