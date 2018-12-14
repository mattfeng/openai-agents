#!/usr/bin/env python

from agent import VanillaPolicyGradientAgent
from observers import StepObserver
from collections import deque
import numpy as np
import tensorflow as tf
import gym

class Experiment():
    def __init__(self,
                 key,
                 hparams,
                 num_epochs,
                 batch_size,
                 render,
                 discount_factor):
        # create the environment
        self.key = key
        self.env = gym.make(self.key)
        print(f"Building environment: {self.key}")

        # create a TF session
        self.sess = tf.Session()

        # create the agent and initialize its variables
        self.hparams = hparams
        self.agent = VanillaPolicyGradientAgent(self.env,
            self.sess, self.hparams)
        self.agent.add_observer(StepObserver(self.agent))
        self.sess.run(tf.global_variables_initializer())

        # define other parameters
        self.num_epochs = num_epochs
        self.render = render
        self.gamma = discount_factor
        self.batch_size = batch_size

        # bookkeeping variables
        self.return_buffer = deque([], maxlen=100)
    
    def run(self):
        for epoch in range(1, self.num_epochs + 1):
            b_states, b_actions, b_advantages = [], [], []
            for rollout in range(1, self.batch_size + 1):
                states, actions, rewards, advantages = self.rollout()

                # prepare training set
                b_states.extend(states)
                b_actions.extend(actions)
                b_advantages.extend(advantages)

                # book keeping
                total_return = sum(rewards)
                self.return_buffer.append(total_return)
                avg_return = sum(self.return_buffer) / len(self.return_buffer)
                print(f"\t [rollout_{rollout}] return: {total_return}, "
                    f"avg_return: {avg_return:.1f}")
            
            b_states = np.array(b_states)
            b_actions = np.array(b_actions)
            b_advantages = np.array(b_advantages)

            loss = self.agent.learn(b_states, b_actions, b_advantages, self.batch_size)
            print(f"[epoch_{epoch}] loss={loss:.4f}")


    def _accumulate(self, rewards):
        advantages = np.zeros_like(rewards)

        acc = 0
        for ix, r in enumerate(rewards[::-1]):
            acc = self.gamma * acc + r
            advantages[len(rewards) - ix - 1] = acc

        advantages -= np.mean(advantages)
        advantages /= np.std(advantages)
        
        return advantages

    def rollout(self):
        s = self.env.reset()
        states, actions, rewards = [], [], []
        done = False

        while not done:
            if self.render:
                self.env.render()
            
            a = self.agent.act(s)
            s_, r, done, _ = self.env.step(a)
            states.append(s)
            actions.append(a)
            rewards.append(r)
            
            s = s_
        
        advantages = self._accumulate(rewards)
        return states, actions, rewards, advantages

if __name__ == "__main__":
    HPARAMS = {
        "learning_rate": 0.03
    }
    NUM_EPOCHS = 100
    BATCH_SIZE = 10
    RENDER = True
    DISCOUNT_FACTOR = 0.99

    exp = Experiment("CartPole-v0",
                     HPARAMS,
                     NUM_EPOCHS,
                     BATCH_SIZE,
                     RENDER,
                     DISCOUNT_FACTOR)
    exp.run()

        
