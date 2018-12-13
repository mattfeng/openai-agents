# Lessons Learned

## Vanilla Policy Gradient
* You can test the shapes and sizes of the network before implementing it into the entire training framework, so that debugging inputs and outputs is something that is completed before the code base becomes more complicated.
* Using observer pattern is very useful.
* Feeding in multiple episodes to gather an unbiased estimate of the gradient helped convergence.

## Deep DQN
* The rate of epsilon decay in the epsilon-greedy policy plays an extremely important role in the convergence of DQNs, particularly because there needs to be a good fraction of "best" actions taken rather than random actions for anything to begin to converge.