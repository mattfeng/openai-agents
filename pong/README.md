# Notes


Make sure the returns are associated correctly! For example, 
in the Pong game, each of the 21 points are independent of each other; thus, the accumulated scores (`cumsum`) for
actions in the trajectory should be reset for each game (i.e.
the move that won in one round didn't cause a loss in
the next round).

The above trick allows the network to train.

Other notes: Establish a baseline of random performance, including
average return and the distribution of scores. That way, you
will know if your model is actually learning anything at all.

Binarize and simplify the inputs to the network as much as possible.
Rewards in RL have low amounts of signal and thus smaller networks
are easier to train.

Encode temporal differences somehow into the state, such as taking
the difference between adjacent frames.