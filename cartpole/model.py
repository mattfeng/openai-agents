import tensorflow as tf
from tfutils.funcs import *

class Agent():
    def __init__(self, M):
        self.M = M
        self.states = tf.placeholder()
        self.actions = tf.placeholder()
        self.discounted_returns = tf.placeholder()

        Linear = tf.contrib.layers.fully_connected
        xavier_init = tf.contrib.layers.xavier_initializer
        xntropy = tf.nn.softmax_cross_entropy_with_logits_v2

        with tf.name_scope("fc1"):
            self.fc1 = Linear(
                inputs=self.states,
                num_outputs=10,
                activation_fn=tf.nn.relu,
                weights_initializer=xavier_init()
            )

        with tf.name_scope("fc2"):
            self.fc2 = Linear(
                inputs=self.fc1,
                num_outputs=5,
                activation_fn=tf.nn.relu,
                weights_initializer=xavier_init()
            )

        with tf.name_scope("fc3"):
            self.fc3 = Linear(
                inputs=self.fc2,
                num_outputs=self.M.env.action_space.n,
                activation_fn=None,
                weights_initializer=xavier_init()
            )

        with tf.name_scope("softmax"):
            self.policy = tf.nn.softmax(logits=self.fc3)
        
        with tf.name_scope("objective"):
            # use a mathematical trick in that the xntropy
            # with the actions will actually select the
            # log probabilities of the actions we took
            # (i.e. a free `gather` function)
            nll = xntropy(
                logits=self.fc3,
                labels=self.actions
            )
            neg_objective = tf.reduce_mean(nll * self.discounted_returns)

        with tf.name_scope("optimize"):
            train_opt = tf.train.AdamOptimizer(learning_rate=M.lr).minimize(
                neg_objective)




