#!/usr/bin/env python

import torch as T
import torch.nn as nn
import gym
import numpy as np

import sys
sys.path.append("/Users/mattfeng/torchutils/")
from torchutils.bootstrap import bootstrap

@bootstrap.main
def main(*args, **kwargs):
    M = kwargs["M"]
    env = gym.make("BreakoutDeterministic-v4")
    frame = env.reset()
    env.render()

    is_done = False
    while not is_done:
        frame, reward, is_done, _ = env.step(env.action_space.sample())
        env.render()

if __name__ == "__main__":
    main()
