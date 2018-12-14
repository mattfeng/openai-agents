import numpy as np
import tensorflow as tf
import gym
from keras.layers import Dense

class VanillaPolicyGradientAgent(object):
    def __init__(self, env, sess, hparams):
        """
            env: OpenAI Gym envrionment
            sess: TensorFlow session
        """
        self.env = env
        self.sess = sess
        self.n_actions = self.env.action_space.n
        self.hp = hparams

        # define the model for the policy network
        self._define_model()
        self.sess.run(tf.global_variables_initializer())

        # define observers
        self.observers = []

    def _define_model(self):
        self.states = tf.placeholder(tf.float32,
            shape=(None, 4), name="Input")
        
        # define the model internals here
        self.dense1 = Dense(16)(self.states)
        self.dense2 = Dense(16)(self.dense1)
        
        self.probs = Dense(self.n_actions,
            activation="softmax")(self.dense2)
        self.log_probs = tf.log(self.probs)

        # training and optimization
        # takes in all the batches together in a single vector
        self.traj_advantages = tf.placeholder(tf.float32)
        self.traj_actions = tf.placeholder(tf.int32)
        self.num_traj = tf.placeholder(tf.float32)

        # turn traj_actions into full row indices
        # tf.shape gets the number actions that were taken
        # (i.e. number of steps in the trajectories)
        # idxs forms the actual indices so that gather_nd
        # can retrieve the values from self.log_probs
        row_idxs = tf.range(tf.shape(self.traj_actions)[0])
        idxs = tf.stack([row_idxs, self.traj_actions], axis=1)

        self.action_logprobs = tf.gather_nd(self.log_probs, idxs)

        # loss = mean(sum(advantage * logprob(actions)))
        # where the mean is taken over different trajectories
        self.loss = -tf.div(
            tf.reduce_sum(self.traj_advantages * self.action_logprobs),
            self.num_traj)

        self.optimizer = tf.train.RMSPropOptimizer(self.hp["learning_rate"])
        self.train = self.optimizer.minimize(self.loss)

    
    def act(self, state):
        # turn singular state into a batch of size 1
        state = np.array([state])
        policy = self.sess.run(self.probs, feed_dict={self.states: state})
        return np.random.choice(np.arange(self.n_actions), p=policy[0])
    
    def learn(self, states, actions, advantages, num_traj):
        loss, _ = self.sess.run(
            [self.loss, self.train],
            feed_dict={
                self.states: states,
                self.traj_actions: actions,
                self.traj_advantages: advantages,
                self.num_traj: num_traj
            }
        )
        return loss
    
    def add_observer(self, obs):
        self.observers.append(obs)
        
    def notify(self, event):
        for obs in self.observers:
            obs(event)

if __name__ == "__main__":
    env = gym.make("CartPole-v0")
    sess = tf.Session()
    hparams = {
        "learning_rate": 0.1
    }
    agent = VanillaPolicyGradientAgent(env, sess, hparams)

    # states = np.array([
    #     [1, 2, 3, 4],
    #     [5, 6, 7, 8]
    # ])
    # log_probs = agent.sess.run(agent.log_probs, feed_dict={agent.states: states})
    # print(log_probs)

    states = np.array([
        [0, 0, 1, 1],
        [0, 1, 1, 1],
        [1, 1, 1, 1]
    ])
    adv = np.array([1, 2, -1])
    actions = np.array([0, 1, 1])

    lp, action_lp, loss = agent.sess.run(
        [agent.log_probs, agent.action_logprobs, agent.loss],
        feed_dict={
            agent.states: states, 
            agent.traj_advantages: adv,
            agent.traj_actions: actions,
            agent.num_traj: 3
        }
    )
    print(lp)
    print()
    print(action_lp)
    print()
    print(f"loss: {loss:.6f}")

    print(agent.act(states[0]))
