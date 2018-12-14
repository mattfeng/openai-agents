#!/usr/bin/env python

from experiment import Experiment

class CartPoleExperiment(Experiment):
    pass

def cartpole():
    HPARAMS = {
        "learning_rate": 0.03
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

def main():
    cartpole()

if __name__ == "__main__":
    main()