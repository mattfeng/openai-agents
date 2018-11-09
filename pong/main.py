#!/usr/bin/env python

import gym
import openai_envs

import tensorflow as tf
from tfutils.viz.display import Display
from tfutils.env import Environment
from tfutils.bootstrap import init_tensorboard

import matplotlib.pyplot as plt

from skimage import data, color
from skimage.transform import resize

from model import Agent

import numpy as np

from collections import deque

import os

DISPLAY_WIDTH = 600
DISPLAY_HEIGHT = 600
DISP = os.environ["DISP"] == "y"

NUM_EPISODES = 10000
GAMMA = 0.99
FRAME_BUFFER_SIZE = 2
OPTIMIZER_OPTIONS = {
    "learning_rate": 1e-3,
    "decay": 0.99
}

def preprocess_state(state):
    state = state[35:195]
    state = state[::2, ::2, 0]
    state[(state == 144) | (state == 109)] = 0
    state[state != 0] = 1
    state = state.reshape((80, 80, 1))
    return state

# def discounted_returns(rewards, normalize=True):
#     """
#     Args:
#         rewards (np.array): Array of rewards at each timestep.
#         normalize (bool): Should the returns be normalized?
#     Returns:
#         Array of discounted returns `G` (sum over discounted rewards).
#     """
#     discounted_g = np.zeros_like(rewards)
#     cumsum = 0
#     for t in range(len(rewards) - 1, -1, -1):
#         cumsum = cumsum * GAMMA + rewards[t]
#         discounted_g[t] = cumsum
    
#     mean = np.mean(discounted_g)
#     stdev = np.std(discounted_g)

#     if normalize:
#         return (discounted_g - mean) / stdev
#     print(discounted_g)
#     return discounted_g

def discounted_returns(r):
    """ take 1D float array of rewards and compute discounted reward """
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(range(0, r.size)):
        if r[t] != 0: running_add = 0 # reset the sum, since this was a game boundary (pong specific!)
        running_add = running_add * GAMMA + r[t]
        discounted_r[t] = running_add
    return discounted_r

def test(M):
    pass

def train(M):
    state = M.env.reset()

    states = []
    actions = []
    rewards = []

    stacked_frames = deque()

    while True:
        # Preprocess state
        if len(stacked_frames) >= FRAME_BUFFER_SIZE:
            stacked_frames.popleft()

        state = preprocess_state(state)
        stacked_frames.append(state)

        if len(stacked_frames) < FRAME_BUFFER_SIZE:
            policy = np.ones(2) / 2
        else:
            diff_state = stacked_frames[1] - stacked_frames[0]

            if DISP:
                M.display.draw_vector(diff_state * 255, 0, 0, scale=2)
            # stacked_states = np.dstack(stacked_frames)
            policy = M.sess.run(M.agent.policy, feed_dict={
                # M.agent.states: stacked_states.reshape([1, 80, 80, 1])
                M.agent.states: diff_state.reshape([1, 80, 80, 1])
            })
            # states.append(stacked_states)
            states.append(diff_state)

        policy = policy.squeeze() # policy.shape: (1, 6) -> (6, )

        action = np.random.choice(
            np.arange(2),
            p=policy)

        next_state, reward, done, _ = M.env.step(action + 2)

        action_onehot = np.zeros_like(policy)
        action_onehot[action] = 1

        if len(stacked_frames) == FRAME_BUFFER_SIZE:
            rewards.append(reward)
            actions.append(action_onehot)

        if done:
            episode_return = np.sum(rewards)
            M.total_return += episode_return
            mean_return = M.total_return / (M.ep + 1)

            neg_obj, _ = M.sess.run([M.agent.neg_obj, M.agent.train_op],
                feed_dict={
                    M.agent.states: np.array(states).reshape([-1, 80, 80, 1]),
                    M.agent.actions: np.array(actions),
                    M.agent.discounted_returns: discounted_returns(np.array(rewards))
                })

            summary = M.sess.run(M.write_op, feed_dict={
                M.agent.neg_obj: neg_obj,
                M.agent.mean_return: M.total_return / (M.ep + 1)
            })

            M.writer.add_summary(summary)
            M.writer.flush()

            return episode_return, mean_return, neg_obj
        
        state = next_state
    

def main():
    M = Environment()
    M.env = gym.make("Pong-v0")
    M.agent = Agent(M, OPTIMIZER_OPTIONS)

    M.writer = init_tensorboard()
    tf.summary.scalar("Negative J(theta)", M.agent.neg_obj)
    tf.summary.scalar("Mean Return", M.agent.mean_return)
    M.write_op = tf.summary.merge_all()

    if DISP:
        M.display = Display("pong", DISPLAY_WIDTH, DISPLAY_HEIGHT)

    # Print basic info about the environment
    print("Action set: {}".format(M.env.unwrapped._action_set))

    M.total_return = 0

    with tf.Session() as sess:
        M.sess = sess
        M.sess.run(tf.global_variables_initializer())

        for ep in range(NUM_EPISODES):
            M.ep = ep
            episode_return, mean_return, neg_obj = train(M)
            print("[ep/{}] G: {:5.2f} meanG: {:5.2f} -J(theta): {:0.12f}".format(
                M.ep, episode_return, mean_return, neg_obj))

    
if __name__ == "__main__":
    main()
