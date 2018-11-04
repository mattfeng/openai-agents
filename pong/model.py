#!/usr/bin/env python

import tensorflow as tf
from tfutils.funcs import *

class Agent(object):
    def __init__(self, M, optimizer_options):
        """
        Agent using REINFORCE algorithm.
        optimizer_options (dict): Options for the optimizer.
        """
        self.M = M

        conv2d = tf.layers.conv2d
        Linear = tf.contrib.layers.fully_connected
        xntropy = tf.nn.softmax_cross_entropy_with_logits_v2

        with tf.name_scope("informative"):
            self.mean_return = tf.placeholder(
                tf.float32,
                name="mean_return")

        with tf.name_scope("inputs"):
            self.states = tf.placeholder(
                tf.float32,
                [None, 84, 84, 1],
                "states")
            self.actions = tf.placeholder(
                tf.int32,
                [None, self.M.env.action_space.n],
                "actions")
            self.discounted_returns = tf.placeholder(
                tf.float32,
                [None,],
                "discounted_returns")

        with tf.name_scope("model"):
            with tf.name_scope("conv1"):
                self.conv1 = conv2d(
                    self.states,
                    filters=128,
                    kernel_size=8,
                    padding="same",
                    strides=4
                )

            with tf.name_scope("conv2"):
                self.conv2 = conv2d(
                    self.conv1,
                    filters=256,
                    kernel_size=4,
                    padding="same",
                    strides=3
                )

            with tf.name_scope("conv3"):
                self.conv3 = conv2d(
                    self.conv2,
                    filters=64,
                    kernel_size=3,
                    padding="same",
                    strides=1
                )

            with tf.name_scope("fc1"):
                self.conv_features = self.conv3.reshape([-1, 7 * 7 * 64])
                self.fc1 = Linear(
                    self.conv_features,
                    num_outputs=40
                )

            with tf.name_scope("fc2"):
                self.fc2 = Linear(
                    self.fc1,
                    num_outputs=self.M.env.action_space.n
                )
            
            with tf.name_scope("softmax"):
                self.policy = tf.nn.softmax(logits=self.fc2)

        with tf.name_scope("objective"):
            nll = xntropy(
                logits=self.fc2,
                labels=self.actions
            )
            self.neg_obj = tf.reduce_mean(nll * self.discounted_returns)

        with tf.name_scope("optimizer"):
            self.train_op = tf.train.GradientDescentOptimizer(
                **optimizer_options).minimize(self.neg_obj)


if __name__ == "__main__":
    import gym
    from tfutils.env import Environment
    from tfutils.preprocess import rgb_to_gray

    M = Environment()
    M.env = gym.make("Pong-v0")
    state = M.env.reset()
    state = rgb_to_gray(state)
    print(state.shape)

    M.agent = Agent(M, {
        "learning_rate": 0.01
    })

    with tf.Session() as sess:
        M.sess = sess
        M.sess.run(tf.global_variables_initializer())
        conv3 = M.sess.run(M.agent.conv3, feed_dict={
            M.agent.states: state.reshape([-1, 210, 160, 1])
        })
        print(conv3.shape)