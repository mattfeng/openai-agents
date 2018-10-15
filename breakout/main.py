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

DISPLAY_ENABLED = os.environ['DISP'] == 'Y'

DISPLAY_WIDTH = 600
DISPLAY_HEIGHT = 600

EPOCHS = 100
BATCH_SIZE = 128
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 2000
TARGET_UPDATE = 10

transform = V.Compose([
    V.ToPILImage(),
    V.Resize((84, 84)),
    V.ToTensor()
])

def optimize_model(M):
    if len(M.memory) < BATCH_SIZE:
        return

    transitions = M.memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))

    non_final_mask = T.tensor(
        tuple(map(lambda s: s is not None, batch.next_state)),
        device=M.device, dtype=T.uint8)

    non_final_next_states = T.cat(
        [s for s in batch.next_state if s is not None])
    state_batch = T.cat(batch.state)
    action_batch = T.cat(batch.action)
    reward_batch = T.cat(batch.reward)

    non_final_next_states = non_final_next_states.to(M.device)
    state_batch = state_batch.to(M.device)
    action_batch = action_batch.to(M.device)
    reward_batch = reward_batch.to(M.device)

    # - examples on the 0th axis, values on the 1st axis
    # - get the output of our model -- what does it believe
    #   the value is for the states we're going to be
    #   transitioning to (or we believe we will be 
    #   transitioning to -- in other words, what is the
    #   predicted value of our action, given our state)?
    state_action_values = M.policy(state_batch).gather(1, action_batch.view(-1, 1))

    # - compute the "actual" value of the next state
    #   we ended up in by taking the above action.
    next_state_values = T.zeros(BATCH_SIZE, device=M.device)
    next_state_values[non_final_mask] = \
        M.target(non_final_next_states).max(dim=1)[0].detach()

    # Compute the expected Q values
    expected_state_action_values = reward_batch + (GAMMA * next_state_values)

    loss = F.smooth_l1_loss(
        state_action_values,
        expected_state_action_values.unsqueeze(1))

    # Optimize the model
    M.optim().zero_grad()
    loss.backward()
    for param in M.policy.parameters():
        param.grad.data.clamp_(-1, 1)
    M.optim().step()

    if M.steps % 20 == 0:
        print("[x] finish optimizing: step {}".format(M.steps))

def train(M):
    print("[*] -- training mode --")

    duration = 0
    env = M.env
    prev_frame = transform(env.reset())
    frame, _, _, _ = env.step(env.action_space.sample())
    frame = transform(frame)
    state = T.cat([frame, prev_frame], dim=0)
    done = False
    M.policy.train()

    for t in count():
        eps = EPS_END + (EPS_START - EPS_END) * \
            math.exp(-1. * M.steps / EPS_DECAY)

        state = state.to(M.device)
        action, was_random  = rl.epsilon_greedy(
            env.action_space.n, state, M.policy, eps)
        
        prev_frame = T.tensor(frame)
        frame, reward, done, _ = env.step(action)
        frame = transform(frame)
        reward = T.tensor([reward], device=M.device)

        if DISPLAY_ENABLED:
            M.display.draw_pytorch_tensor(frame, 0, 0)
            action_label = "[i] action: {}".format(M.action_db[action])
            M.display.draw_text(action_label, 10, DISPLAY_HEIGHT - 30)
            eps_label = "[i] eps: {:0.2f} (random? {})".format(eps, was_random)
            M.display.draw_text(eps_label, 10, DISPLAY_HEIGHT - 70)

        if done:
            next_state = None
        else:
            next_state = T.cat([frame, prev_frame], dim=0)
        
        M.memory.push(state, T.tensor([action]), next_state, reward)

        state = next_state
        M.steps += 1

        optimize_model(M)

        if done:
            duration = t + 1
            break

    # Update the target network
    if M.epoch % TARGET_UPDATE == 0:
        M.target.load_state_dict(M.policy.state_dict())
    
    return duration

def test(M):
    print("[*] -- testing mode --")
    env = M.env
    prev_frame = transform(env.reset())
    frame, _, _, _ = env.step(env.action_space.sample())
    frame = transform(frame)
    done = False

    with T.no_grad():
        t = 0
        while not done:
            state = T.cat([frame, prev_frame], dim=0)
            state = state.to(M.device)

            eps = 0.1
            action, was_random = rl.epsilon_greedy(
                M.env.action_space.n, state, M.policy, eps)

            prev_frame = T.tensor(frame)
            frame, _, done, _ = env.step(action)

            frame = transform(frame)

            if DISPLAY_ENABLED:
                M.display.draw_pytorch_tensor(frame, 0, 0)
                action_label = "[i] action: {}".format(M.action_db[action])
                M.display.draw_text(action_label, 10, DISPLAY_HEIGHT - 30)
            
            t += 1
    return t
    


@bootstrap.main
def main(*args, **kwargs):
    M = kwargs["M"]
    M.env = gym.make("BreakoutDeterministic-v4")

    M.policy = DQN()
    M.target = DQN()
    M.target.eval()
    M.policy.to(M.device)
    M.target.to(M.device)

    M.memory = rl.ReplayMemory(10000)
    if DISPLAY_ENABLED:
        M.display = Display("breakout", DISPLAY_WIDTH, DISPLAY_HEIGHT)
    M.action_db = {
        0: "NOP",
        1: "Fire",
        2: "Right",
        3: "Left"
    }

    M.optim(optim.RMSprop(M.policy.parameters()))
    M.steps = 0

    durations = []
    test_durations = []
    for epoch in range(EPOCHS):
        M.epoch = epoch
        duration = train(M)
        durations.append(duration)
        print("[train/{}] duration: {}, total steps: {}".format(
            epoch, duration, M.steps))
        test_duration = test(M)
        test_durations.append(test_duration)
        print("[test/{}] test_duration: {}".format(epoch, test_duration))


if __name__ == "__main__":
    main()
