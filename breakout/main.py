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

DISPLAY_WIDTH = 600
DISPLAY_HEIGHT = 600

@bootstrap.main
def main(*args, **kwargs):
    M = kwargs["M"]
    M.display = Display(DISPLAY_WIDTH, DISPLAY_HEIGHT)

    env = gym.make("BreakoutDeterministic-v4")
    frame = env.reset()
    env.render()

    is_done = False
    transform = V.Compose([
        V.ToPILImage(),
        V.Resize((160, 160)),
        V.ToTensor()
    ])

    while not is_done:
        frame, reward, is_done, _ = env.step(env.action_space.sample())
        frame = transform(frame)

        M.display.draw_torchvision(frame, 0, 0)

        env.render()
        time.sleep(0.1)

if __name__ == "__main__":
    main()
