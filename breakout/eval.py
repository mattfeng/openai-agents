#!/usr/bin/env python

import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as V

import gym
import numpy as np
import matplotlib.pyplot as plt
import time
import math
from itertools import count

import os
import sys
sys.path.append("{}/torchutils/".format(os.environ["HOME"]))
from torchutils.bootstrap import bootstrap
from torchutils.viz.display import Display
import torchutils.models.rl as rl

from model import DQN, Transition

DISPLAY_WIDTH = 600
DISPLAY_HEIGHT = 600

transform = V.Compose([
    V.ToPILImage(),
    V.Resize((84, 84)),
    V.ToTensor()
])

def eval(M):
    print("[*] -- evaluation mode --")
    env = M.env
    prev_frame = transform(env.reset())
    frame, _, _, _ = env.step(1)
    frame = transform(frame)
    done = False
    consecutive_same = 0

    with T.no_grad():
        t = 0
        while not done:
            state = T.cat([frame, prev_frame], dim=0)
            state = state.to(M.device)

            eps = 0.0
            action, was_random, _ = rl.epsilon_greedy(
                M.env.action_space.n, state, M.policy, eps)
                
            if consecutive_same > 30:
                print("[i] action overriden")
                action = 1
                consecutive_same = 0

            prev_frame = T.tensor(frame)
            frame, _, done, _ = env.step(action)

            frame = transform(frame)

            same = T.all(T.lt(
                T.abs(T.add(prev_frame[:-10, :], -frame[:-10, ])), 1e-8)).item()

            if same == 0:
                consecutive_same = 0
            else:
                consecutive_same += 1


            M.display.draw_pytorch_tensor(frame, 0, 0)
            action_label = "[i] action: {} (random? {})".format(
                M.action_db[action], was_random)
            M.display.draw_text(action_label, 10, DISPLAY_HEIGHT - 30)
            
            t += 1
    return t


@bootstrap.main
def main(*args, **kwargs):
    model_file = "model-epoch-170-time-1539656585.pt"
    M = kwargs["M"]
    M.env = gym.make("Breakout-v4")

    M.policy = DQN()
    M.policy.load_state_dict(
        T.load("./models/{}".format(model_file), map_location=M.device))

    M.display = Display("breakout", DISPLAY_WIDTH, DISPLAY_HEIGHT)

    M.action_db = {
        0: "NOP",
        1: "Fire",
        2: "Right",
        3: "Left"
    }

    duration = eval(M)
    print("[i] duration: {}".format(duration))

if __name__ == "__main__":
    main()