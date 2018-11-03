#!/usr/bin/env python

import gym

import tensorflow as tf
from tfutils.env import Environment
from tfutils.bootstrap import init_tensorboard

from model import Agent

import numpy as np

LEARNING_RATE = 0.01
GAMMA = 0.99
NUM_EPISODES = 10000
MAX_STEPS = 999

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

    return (discounted_g - mean) / stdev

def train(M):
    episode_states = []
    episode_actions = []
    episode_rewards = []
    
    # Launch the game
    state = M.env.reset()
        
    M.env.render()
           
    while True:
        policy = M.sess.run(M.agent.policy, feed_dict={
            M.agent.states: state.reshape([1, 4])
        })
            
        # select action w.r.t the actions prob
        action = np.random.choice(range(policy.shape[1]), p=policy.ravel())

        # Perform a
        new_state, reward, done, info = M.env.step(action)

        # Store s, a, r
        episode_states.append(state)
        action_onehot = np.zeros(M.env.action_space.n)
        action_onehot[action] = 1
        episode_actions.append(action_onehot)
        episode_rewards.append(reward)

        if done:
            # Calculate return
            episode_return = np.sum(episode_rewards)
            M.all_returns.append(episode_return)
            M.total_returns += episode_return

            # Mean return
            mean_return = np.divide(M.total_returns, M.episode + 1)

            # Maximum return
            M.max_return = np.amax(M.all_returns)
                
            # Calculate discounted returns
            discounted_g = discounted_returns(episode_rewards)
                                
            # Feedforward, gradient and backpropagation
            neg_obj, _ = M.sess.run([M.agent.neg_objective, M.agent.train_op], feed_dict={
                M.agent.states: np.vstack(np.array(episode_states)),
                M.agent.actions: np.vstack(np.array(episode_actions)),
                M.agent.discounted_returns: discounted_g 
            })
                                                                 
            # Write TF Summaries
            summary = M.sess.run(M.write_op, feed_dict={
                M.agent.states: np.vstack(np.array(episode_states)),
                M.agent.actions: np.vstack(np.array(episode_actions)),
                M.agent.discounted_returns: discounted_g,
                M.agent.mean_return: mean_return
            })
                
            M.writer.add_summary(summary, M.episode)
            M.writer.flush()

            return episode_return, mean_return
            
        state = new_state


def main():
    M = Environment()
    M.env = gym.make("CartPole-v0")
    M.env.state_size = 4

    # Hyperparameters
    M.lr = LEARNING_RATE

    M.agent = Agent(M)
    M.writer = init_tensorboard()
    tf.summary.scalar("Negative J(theta)", M.agent.neg_objective)
    tf.summary.scalar("Mean Return", M.agent.mean_return)
    M.write_op = tf.summary.merge_all()

    M.all_returns = []
    M.total_returns = 0
    M.max_return = 0

    
    with tf.Session() as sess:
        M.sess = sess
        M.sess.run(tf.global_variables_initializer())

        for ep in range(NUM_EPISODES):
            M.episode = ep
            ep_return, mean_return =  train(M)

            print("[ep/{}] G: {:>5.0f} mean_G: {:>5.0f} "
                  "max_G: {:>5.0f}".format(
                      M.episode,
                      ep_return,
                      mean_return,
                      M.max_return))

if __name__ == "__main__":
    main()