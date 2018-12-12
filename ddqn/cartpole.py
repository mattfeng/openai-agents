#!/usr/bin/env python

from agent import DDQNAgent
from itertools import count
from collections import deque
from keras.layers import Dense
from params import HIDDEN_SIZE
import gym

NUM_EPISODES = 10000

def define_model():
    return [
        Dense(HIDDEN_SIZE, activation="relu"),
        Dense(HIDDEN_SIZE, activation="relu")
    ]

def main():
    env = gym.make("CartPole-v0")
    agent = DDQNAgent(env, define_model())
    rewards = deque([], maxlen=100)

    for episode in range(1, NUM_EPISODES + 1):
        s = env.reset()
        done = False
        reward = 0

        for j in count(1):
            env.render()
            a = agent.act(s)
            s_, r, done, _ = env.step(a)
            reward += r
            loss = agent.learn((s, a, r, s_, done))

            if done:
                rewards.append(reward)
                avg_return = sum(rewards) / len(rewards)
                print(f"[ep{episode}] "
                      f"avg return: {avg_return:.2f}, "
                      f"current ret: {reward:.2f}, "
                      f"eps: {agent.eps:.3f}")
                break

            s = s_



if __name__ == "__main__":
    main()