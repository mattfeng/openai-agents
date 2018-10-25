#!/usr/bin/env python

import gym

import os
import sys
sys.path.append("{}/torchutils/".format(os.environ["HOME"]))
from torchutils.bootstrap import bootstrap
from torchutils.viz.display import Display
import torchutils.models.rl as rl
import torchutils.train as tr

import time
from itertools import count

import torch as T
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as V

import numpy as np

from model import DuelingDQN, DQN

transform = V.Compose([
    V.ToPILImage(),
    V.Grayscale(),
    V.Resize((84, 84)),
    V.ToTensor()
])

disp_transform = V.Compose([
    V.ToTensor()
])

# Runtime variables
DISPLAY_ENABLED = os.environ["DISP"].lower() == "y"
DISPLAY_WIDTH = 600
DISPLAY_HEIGHT = 600

# Hyperparameters
EPOCHS = 10000
EPS_START = 1.00
EPS_END = 0.1
EPS_DECAY = 1e6
STEPS_BEFORE_TRAIN = 25000
BATCH_SIZE = 32
TARGET_UPDATE = 10

REPLAY_BUF_SIZE = 30000
GAMMA = 0.99 # decay rate

LEARNING_RATE = 0.000075

def test(M):
    M.log("begin TESTING")

    env = M.env
    M.env.reset()

    prev_frame = transform(env.reset())
    frame, _, _, _ = env.step(0)
    frame = transform(frame)

    done = False
    cum_reward = 0

    with T.no_grad():

        for duration in count(1):
            state = T.cat([frame, prev_frame], dim=0)
            state = state.to(M.device)

            eps = 0.0
            action, was_random, action_values = rl.epsilon_greedy(
                M.env.action_space.n, state, M.policy, eps)
            
            if was_random:
                action = 0

            if duration % 200 == 0:
                print("action values: {}".format(action_values))

            prev_frame = T.tensor(frame)
            frame, reward, done, _ = env.step(action)
            cum_reward += reward
            frame = transform(frame)

            action_label = "[i] action: {}".format(M.action_db[action])
            if DISPLAY_ENABLED:
                M.display.draw_pytorch_tensor(frame, 0, 0)
                M.display.draw_text(action_label, 10, DISPLAY_HEIGHT - 30)
            
            if done:
                return cum_reward, duration

def train(M):
    M.log("begin TRAINING mode")

    # Set the network to training mode
    M.policy.train()
    env = M.env

    # Reset the environment
    prev_frame = transform(env.reset())
    frame, _, _, _ = env.step(0)
    frame = transform(frame)
    
    # Create the initial state
    state = T.cat([frame, prev_frame], dim=0)

    # Keep track of loss and duration
    total_loss = 0
    num_loss = 1
    done = False
    cum_reward = 0

    for duration in count(1):
        # Decrease the chance of random action as training progresses
        eps = tr.eps(EPS_START, EPS_END, EPS_DECAY,
            M.steps, offset=STEPS_BEFORE_TRAIN)
        M.eps = eps

        # Compute an action using the epsilon greedy policy
        state = state.to(M.device)
        action, was_random, action_values  = rl.epsilon_greedy(
            env.action_space.n, state, M.policy, eps)

        prev_frame = T.tensor(frame)
        frame, reward, done, _ = env.step(action)
        if reward > 0:
            print("+reward (raw):", reward)

        disp_frame = disp_transform(frame)
        frame = transform(frame)
        reward = T.tensor([float(np.sign(int(reward)))], device=M.device)

        cum_reward += reward.item()

        if DISPLAY_ENABLED:
            M.display.draw_pytorch_tensor(disp_frame, 0, 0, scale=2)
            M.display.draw_pytorch_tensor(frame, 330, 0, scale=2)
            action_label = "[i] action: {}".format(M.action_db[action])
            M.display.draw_text(action_label, 10, DISPLAY_HEIGHT - 30)
            eps_label = "[i] eps: {:0.2f} (random? {})".format(eps, was_random)
            M.display.draw_text(eps_label, 10, DISPLAY_HEIGHT - 60)
            reward_label = "[i] reward: {:0.2f} (cum: {:0.2f})".format(reward.item(), cum_reward)
            M.display.draw_text(reward_label, 10, DISPLAY_HEIGHT - 90)
        else:
            reward_label = "[i] reward: {}".format(reward.item())

        if done:
            next_state = None
        else:
            next_state = T.cat([frame, prev_frame], dim=0)
        
        # Add the transition to the replay memory
        M.memory.push(state, T.tensor([action]), next_state, reward)

        state = next_state
        M.steps += 1

        # Update M.policy with DDQN Q-learning algorithm
        if M.steps > STEPS_BEFORE_TRAIN and len(M.memory) > BATCH_SIZE:
            loss = optimize(M)
            total_loss += loss
            num_loss += 1

        if done:
            print()
            break

    # Update the target network
    if M.epoch % TARGET_UPDATE == 0:
        M.target.load_state_dict(M.policy.state_dict())
    
    return cum_reward, duration, total_loss / num_loss


def optimize(M):
    # Sample transitions from the Replay Memory
    transitions = M.memory.sample(BATCH_SIZE)
    batch = rl.Transition(*zip(*transitions))

    # Reward for terminating states is 0
    non_final_mask = T.tensor(
        tuple(map(lambda s: s is not None, batch.next_state)),
        device=M.device, dtype=T.uint8)

    # Transpose transitions for PyTorch (i.e. examples along axis 0)
    non_final_next_states = T.cat(
        [s.unsqueeze(0) for s in batch.next_state if s is not None], dim=0)
    state_batch = T.cat(batch.state)
    action_batch = T.cat(batch.action)
    reward_batch = T.cat(batch.reward)

    # Move to corresponding device
    non_final_next_states = non_final_next_states.to(M.device)
    state_batch = state_batch.to(M.device)
    action_batch = action_batch.to(M.device)
    reward_batch = reward_batch.to(M.device)

    # Get the values of Q(s, a) for the actions that we actually took
    state_action_values = M.policy(state_batch).gather(1, action_batch.view(-1, 1))

    next_state_values = T.zeros(BATCH_SIZE, device=M.device)

    # next_state_values[non_final_mask] = \
    #     M.target(non_final_next_states).max(dim=1)[0].detach()

    # Implement Double Q-Learning
    est_best_actions = M.policy(non_final_next_states).argmax(dim=1).view(-1, 1)
    next_state_values[non_final_mask] = M.target(
        non_final_next_states).detach().gather(1, est_best_actions).squeeze()

    # Update only those Q(s, a) where we actually took action a
    expected_state_action_values = reward_batch + (GAMMA * next_state_values)

    loss = F.smooth_l1_loss(
        state_action_values,
        expected_state_action_values.unsqueeze(1))
    
    # Optimize the model
    M.optim().zero_grad()
    loss.backward()
    for param in M.policy.parameters():
        print(param.grad.data)
        param.grad.data.clamp_(-10, 10) # Clip gradients
    M.optim().step()

    if M.steps % 20 == 0:
        print(".", end="", flush=True)
    
    return loss


@bootstrap.main
def main(*args, **kwargs):
    M = kwargs["M"]

    M.env = gym.make("Enduro-v0")

    # Print general information about the environment
    print(M.env.action_space)
    print(M.env.unwrapped._action_set)

    M.action_db = {
        0: "NOP",
        1: "Fire", # Speed up
        2: "Right",
        3: "Left",
        4: "Down",
        5: "Down-Right",
        6: "Down-Left",
        7: "Right-Fire",
        8: "Left-Fire"
    }

    # Initialize display
    if DISPLAY_ENABLED:
        M.display = Display("enduro", DISPLAY_WIDTH, DISPLAY_HEIGHT)

    # Initialize models, and ring buffer
    M.policy = DQN(2, M.env.action_space.n)
    M.target = DQN(2, M.env.action_space.n)
    M.target.eval()

    # Move networks to the corresponding device
    M.policy.to(M.device)
    M.target.to(M.device)

    M.memory = rl.ReplayMemory(REPLAY_BUF_SIZE)

    # Define optimizer
    M.steps = 0
    M.optim(optim.RMSprop(M.policy.parameters(), lr=LEARNING_RATE))


    for M.epoch in range(EPOCHS):
        reward, duration, avg_loss = train(M)
        M.log("[train/{}] reward={:.4f} duration={:.0f} avg_loss={:0.6f} eps={:0.3f}".format(
            M.epoch, reward, duration, avg_loss, M.eps
        ))

        if M.steps >= STEPS_BEFORE_TRAIN:
            reward, duration = test(M)
            M.data("[test/{}] reward={:.4f} duration={:.0f}".format(
                M.epoch, reward, duration
            ))

if __name__ == "__main__":
    main()
