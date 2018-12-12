from agent import VanillaPolicyGradientAgent
from observers import StepObserver
import tensorflow as tf
import gym

class Experiment():
    def __init__(self, key):
        self.key = key
        self.env = gym.make(self.key)
        print(f"Building environment: {self.key}")
        self.sess = tf.Session()

        self.agent = VanillaPolicyGradientAgent(self.sess)
        self.agent.add_observer(StepObserver(self.agent))

        self.sess.run(tf.global_variables_initializer())

    def rollout(self):
        self.env.reset()
        self
        pass
