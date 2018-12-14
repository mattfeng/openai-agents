#!/usr/bin/env python

from cartpole import CartPoleExperiment
from pong import PongExperiment

def cartpole():
    HPARAMS = {
        "learning_rate": 0.03,
        "hidden_size": 16,
        "decay_rate": 0.9
    }
    NUM_EPOCHS = 100
    BATCH_SIZE = 10
    RENDER = True
    DISCOUNT_FACTOR = 0.99
    exp = CartPoleExperiment(
            "CartPole-v0",
            HPARAMS,
            NUM_EPOCHS,
            BATCH_SIZE,
            RENDER,
            DISCOUNT_FACTOR)
    exp.run()

def pong():
    HPARAMS = {
        "learning_rate": 0.0001,
        "hidden_size": 200,
        "decay_rate": 0.99
    }
    NUM_EPOCHS = 10000
    BATCH_SIZE = 10
    RENDER = False
    DISCOUNT_FACTOR = 0.99
    exp = PongExperiment(
            "PongDeterministic-v0",
            HPARAMS,
            NUM_EPOCHS,
            BATCH_SIZE,
            RENDER,
            DISCOUNT_FACTOR)
    exp.run()

def main():
    # cartpole()
    pong()

if __name__ == "__main__":
    main()