# Breakout

## About
* This is an implementation of the Deep Q-Network introduced by Google DeepMind using PyTorch.
The network consists of three convolutional layers and two feed-forward layers, with experience replay for stabilized training.
One of the most novel features of the papers is the training method: using a cache of transitions in order to decouple action values with time.

## Required dependencies
* `torchutils`


## Running the model
```bash
DISP=n ./main.py # no dashboard display
DISP=Y ./main.py # with dashboard display
```
