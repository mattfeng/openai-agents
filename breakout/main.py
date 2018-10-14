#!/usr/bin/env python

import torch as T
import torch.nn as nn
import torchvision.transforms as V
import gym
import numpy as np
import matplotlib.pyplot as plt
import time

import sys
sys.path.append("/Users/mattfeng/torchutils/")
from torchutils.bootstrap import bootstrap
from torchutils.viz.display import Display
import torchutils.models.rl as rl

DISPLAY_WIDTH = 600
DISPLAY_HEIGHT = 600

transform = V.Compose([
    V.ToPILImage(),
    V.Resize((160, 160)),
    V.ToTensor()
])

def train(M):
    pass

def test(M):
    env = M.env
    prev_frame = transform(env.reset())
    frame, _, _, _ = env.step(env.action_space.sample())
    frame = transform(frame)
    done = False

    while not done:
        state = T.stack([frame, prev_frame], dim=2)

        print(env.action_space.sample())

        eps = 0.1
        action = rl.epsilon_greedy(
            M.env.action_space.n, state, M.model, eps)

        prev_frame = T.tensor(frame)
        frame, reward, done, _ = env.step(action)
        frame = transform(frame)

        M.display.draw_torchvision(frame, 0, 0)

        env.render()
        time.sleep(0.1)

@bootstrap.main
def main(*args, **kwargs):
    M = kwargs["M"]
    M.display = Display(DISPLAY_WIDTH, DISPLAY_HEIGHT)
    M.env = gym.make("BreakoutDeterministic-v4")
    M.model = lambda x: np.random.randint(4)

    test(M)

if __name__ == "__main__":
    main()
