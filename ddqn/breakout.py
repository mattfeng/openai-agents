#!/usr/bin/env python

from agent import DDQNAgent
from itertools import count
from collections import deque
from keras.layers import Dense, Conv2D, Flatten
import os
import numpy as np
import gym

RENDER = os.environ["RENDER"] == "Y"
NUM_EPISODES = 10000

def define_model():
    return [
        Conv2D(16, (8, 8), padding="same"),
        Conv2D(32, (4, 4), padding="same"),
        Conv2D(32, (3, 3), padding="same"),
        Flatten(),
        Dense(32, activation="relu"),
        Dense(32, activation="relu")
    ]

def preprocess(s, s_):
    # convert to grayscale
    s = np.dot(s[..., :3], [0.299, 0.587, 0.114])
    s_ = np.dot(s_[..., :3], [0.299, 0.587, 0.114])
    s = np.expand_dims(s, -1)
    s_ = np.expand_dims(s_, -1)

    return s_ - s

def main():
    env = gym.make("BreakoutDeterministic-v0")
    agent = DDQNAgent(env, define_model(), (210, 160, 1))
    rewards = deque([], maxlen=100)

    for episode in range(1, NUM_EPISODES + 1):
        s = env.reset()
        s_, _, _, _ = env.step(1)

        # need to take the difference of frames
        processed_s = preprocess(s, s_)

        prev_num_lives = 5
        press_fire = False

        done = False
        reward = 0

        for j in count(1):
            if RENDER:
                env.render()

            a = agent.act(processed_s)

            if press_fire:
                a = 1
                press_fire = False

            s_, r, done, info = env.step(a)
            num_lives = info["ale.lives"]

            if num_lives < prev_num_lives:
                press_fire = True

            reward += r

            # preprocess states
            processed_s_ = preprocess(s, s_)

            loss = agent.learn((processed_s, a, r, processed_s_, done))

            if done:
                rewards.append(reward)
                avg_return = sum(rewards) / len(rewards)
                print(f"[ep{episode}] "
                      f"avg return: {avg_return:.2f}, "
                      f"current ret: {reward:.2f}, "
                      f"eps: {agent.eps:.3f}")
                break

            prev_num_lives = num_lives
            processed_s = processed_s_
            s = s_


if __name__ == "__main__":
    main()