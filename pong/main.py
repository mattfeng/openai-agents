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

import os

DISPLAY_WIDTH = 600
DISPLAY_HEIGHT = 600
DISP = os.environ["DISP"] == "y"

NUM_EPISODES = 10000
GAMMA = 0.99
OPTIMIZER_OPTIONS = {
    "learning_rate": 0.001
}

def preprocess_state(state):
    state = color.rgb2gray(state)
    resized = resize(state, (84, 84, 1), anti_aliasing=False)
    normalized = resized / 255.
    return normalized

def discounted_returns(rewards, normalize=True):
    """
    Args:
        rewards (np.array): Array of rewards at each timestep.
        normalize (bool): Should the returns be normalized?
    Returns:
        Array of discounted returns `G` (sum over discounted rewards).
    """
    discounted_g = np.zeros_like(rewards)
    cumsum = 0
    for t in range(len(rewards) - 1, -1, -1):
        cumsum = cumsum * GAMMA + rewards[t]
        discounted_g[t] = cumsum
    
    mean = np.mean(discounted_g)
    stdev = np.std(discounted_g)

    if normalize:
        return (discounted_g - mean) / stdev
    return discounted_g

def test(M):
    pass

def train(M):
    state = M.env.reset()

    states = []
    actions = []
    rewards = []

    while True:
        # Preprocess state
        state = preprocess_state(state)
        policy = M.sess.run(M.agent.policy, feed_dict={
            M.agent.states: state.reshape([1, 84, 84, 1])
        })
        policy = policy.squeeze() # policy.shape: (1, 6) -> (6, )

        action = np.random.choice(
            np.arange(M.env.action_space.n),
            p=policy)

        next_state, reward, done, _ = M.env.step(action)

        states.append(state)
        rewards.append(reward)

        action_onehot = np.zeros_like(policy)
        action_onehot[action] = 1

        actions.append(action_onehot)

        M.env.render()
        if DISP:
            M.display.draw_vector(state, 0, 0, scale=2)

        if done:
            episode_return = np.sum(rewards)
            M.total_return += episode_return
            mean_return = M.total_return / (M.ep + 1)
            np_states = np.array(states)

            neg_obj, _ = M.sess.run([M.agent.neg_obj, M.agent.train_op],
                feed_dict={
                    M.agent.states: np.array(states),
                    M.agent.actions: np.array(actions),
                    M.agent.discounted_returns: discounted_returns(rewards)
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
