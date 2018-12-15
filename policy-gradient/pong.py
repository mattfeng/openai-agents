from experiment import Experiment
from agent import VanillaPolicyGradientAgent
from observers import StepObserver
from skimage import color
from skimage.transform import resize
from sklearn.preprocessing import binarize
from collections import deque
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np

class PongExperiment(Experiment):
    def _define_agent(self):
        self.agent = VanillaPolicyGradientAgent(self.env,
            self.sess, self.hparams, input_shape=(6400,))
        self.agent.add_observer(StepObserver(self.agent))

    def _accumulate(self, rewards):
        advantages = np.zeros_like(rewards)

        acc = 0
        for ix, r in enumerate(rewards[::-1]):
            # if you lose a point, don't leave a trace for all the previous moves
            if r != 0:
                acc = 0
            acc = self.gamma * acc + r
            advantages[len(rewards) - ix - 1] = acc


        advantages -= np.mean(advantages)
        advantages /= np.std(advantages)
        
        return advantages
    
    def _preprocess(self, s):
        s[s[:, :] == [109, 118, 43]] = 0
        s[~(s[:, :] == [0, 0, 0])] = 255
        s = s[33:193, :, 0]
        s = s.astype(np.float32)[::2, ::2]
        s /= 255.0
        s -= 0.5
        return s

    def _process(self, s, s_):
        s = self._preprocess(s)
        s_ = self._preprocess(s_)

        # compute the difference, flatten
        diff = s - s_
        diff = np.ravel(diff)

        return diff

    def rollout(self):
        s = self.env.reset()
        s_, _, _, _ = self.env.step(0)

        states, actions, rewards = [], [], []
        done = False

        while not done:
            if self.render:
                self.env.render()
            
            ps = self._process(s, s_)
            a = self.agent.act(ps)

            s = s_
            s_, r, done, _ = self.env.step(a)

            states.append(ps)
            actions.append(a)
            rewards.append(r)
            
        advantages = self._accumulate(rewards)
        return states, actions, rewards, advantages

if __name__ == "__main__":
    import gym
    import matplotlib.pyplot as plt

    env = gym.make("PongDeterministic-v0")
    s = env.reset()
    s[s[:, :] == [109, 118, 43]] = 0
    s[~(s[:, :] == [0, 0, 0])] = 255
    s = s[32:192, :, 0]
    s = resize(s, (84, 84), anti_aliasing=False)
    s /= 255
    s -= 0.5
    
    # s = resize(s, (84, 84), anti_aliasing=False)
    plt.imshow(s)
    plt.show()
