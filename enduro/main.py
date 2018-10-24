#!/usr/bin/env python

import gym

import sys
sys.path.append('/Users/mattfeng/torchutils/')
from torchutils.bootstrap import bootstrap
from torchutils.viz.display import Display

import time
from itertools import count

import torch as T
import torchvision.transforms as V

transform = V.Compose([
    V.ToPILImage(),
    V.Grayscale(),
    V.Resize((84, 84)),
    V.ToTensor()
])

def test():
    pass

def train():
    M.env.reset()
    done = False

def optimize(M):
    pass


@bootstrap.main
def main(*args, **kwargs):
    M = kwargs["M"]

    M.env = gym.make("Enduro-v0")

    # Print general information about the environment
    print(M.env.action_space)
    print(M.env.unwrapped._action_set)

    for t in count(1):
        M.env.render()
        frame, reward, done, info = M.env.step(M.env.action_space.sample())
        if done:
            print("[i] episode finished after {} frames".format(t))
            break
        time.sleep(0.01)

    print("[!] finished running `main.py`")

if __name__ == "__main__":
    main()