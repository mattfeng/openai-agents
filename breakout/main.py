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

from model import DQN

DISPLAY_WIDTH = 600
DISPLAY_HEIGHT = 600

transform = V.Compose([
    V.ToPILImage(),
    V.Resize((84, 84)),
    V.ToTensor()
])

def train(M):
    print("[*] -- training mode --")
    env = M.env
    prev_frame = transform(env.reset())
    frame, _, _, _ = env.step(env.action_space.sample())
    frame = transform(frame)
    done = False

    M.model.train()
    



def test(M):
    print("[*] -- testing mode --")
    env = M.env
    prev_frame = transform(env.reset())
    frame, _, _, _ = env.step(env.action_space.sample())
    frame = transform(frame)
    done = False

    with T.no_grad():
        while not done:
            state = T.cat([frame, prev_frame], dim=0)

            eps = 0.1
            action = rl.epsilon_greedy(
                M.env.action_space.n, state, M.model, eps)

            prev_frame = T.tensor(frame)
            frame, _, done, _ = env.step(action)

            frame = transform(frame)

            M.display.draw_pytorch_tensor(frame, 0, 0)
            action_label = "[i] action: {}".format(M.action_db[action])
            M.display.draw_text(action_label, 10, DISPLAY_HEIGHT - 30)

            time.sleep(0.05)

@bootstrap.main
def main(*args, **kwargs):
    M = kwargs["M"]
    M.env = gym.make("BreakoutDeterministic-v4")
    M.model = DQN()
    M.memory = rl.ReplayMemory(10000)
    M.display = Display("breakout", DISPLAY_WIDTH, DISPLAY_HEIGHT)
    M.action_db = {
        0: "0",
        1: "1",
        2: "Right",
        3: "Left"
    }

    EPOCHS = 100

    for epoch in range(EPOCHS):
        train(M)
        test(M)

if __name__ == "__main__":
    main()
