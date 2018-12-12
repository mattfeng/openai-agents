import math
import numpy as np
import random
from model import DDQN
from memory import ReplayMemory
from params import *

class DDQNAgent(object):
    """
    Defines a double dueling reinforcement learning agent.
    """

    def __init__(self, env, layers, state_shape):
        self.layers = layers
        self.global_step = 1
        self.is_learning = False
        self.training_step_delay = TRAINING_STEP_DELAY
        self.eps = EPSILON_START
        self.eps_start = EPSILON_START
        self.eps_decay = EPSILON_DECAY
        self.eps_end = EPSILON_END
        self.gamma = DISCOUNT_FACTOR
        self.batch_size = TRAINING_SAMPLE_SIZE
        self.target_update_rate = TARGET_UPDATE_RATE
        self.lr = LEARNING_RATE
        
        self.env = env
        self.n_actions = self.env.action_space.n
        # self.state_shape = self.env.observation_space.shape
        self.state_shape = state_shape
        self.target_nn = self.create_ddqn_model()
        self.nn = self.create_ddqn_model()
        self.memory = ReplayMemory(REPLAY_BUFFER_SIZE)
        self.observers = []
    
    def create_ddqn_model(self):
        return DDQN(self.state_shape,
                    self.n_actions,
                    self.layers,
                    self.lr)
    
    def act(self, state):
        # estimate the Q value of the actions
        # based on the current state
        best_action = np.argmax(self.nn.estimate_q(np.array([state])))

        # use epsilon greedy policy
        if random.random() < self.eps:
            action = random.randrange(0, self.n_actions)
        else:
            action = best_action
        
        return action
        
    
    def learn(self, sarsd):
        """
        Args:
            sarsd: (state, action, reward, next_state, done)
        
        Returns:
            None
        """
        self.memory.store([sarsd])
        loss = None
        if self.is_learning:
            # make training batch, update current network
            X, y = self._make_training_batch()
            result = self.nn.fit(X, y)
            loss = result.history["loss"]

            if self.global_step % self.target_update_rate == 0:
                print("updating network")
                self.update_target_network()

        self.notify("finish_step")
        return loss
    
    def notify(self, event):
        for obs in self.observers:
            obs(event)
    
    def finish_step(self):
        if self.global_step > self.training_step_delay:
            self.is_learning = True
        self.global_step += 1

        self.eps = (self.eps_end + (self.eps_start - self.eps_end) *
            math.exp(-self.eps_decay * self.global_step))

    def update_target_network(self):
        weights = self.nn.get_model_weights()
        self.target_nn.set_model_weights(weights)
        self.nn.save_weights("weights.h5")

    def _make_training_batch(self):
        """
        Returns:
            A training batch (X, y), where X consists of a list of
            states and y is the corresponding vector of target
            values for all actions in each state.

            The training error is MSE, which is (target - Q(s))^2.
            Q(s) returns a vector of Q values for
            all the actions.

            Since our neural network outputs Q(s, a) values
            for all the actions, but we only can compute
            the TD error for the specific action a' that was
            taken, we set the value of the target for all
            other actions to be the same as our current
            prediction network, so that the error is 0.
        """
        # sample experiences from the replay buffer
        exps = self.memory.sample(self.batch_size)

        # X, y to use to fit the model later on
        states = []
        targets = []
        for state, action, reward, next_state, done in exps:
            states.append(state)

            # predict values of Q(state, *)
            # for all the different actions
            target = self.nn.estimate_q(np.array([state])).squeeze()
            # the value of the action that was taken
            # is the reward, plus the Q value
            # best action in the next state
            best_action = np.argmax(self.nn.estimate_q(np.array([next_state])).squeeze())
            # predict value of the best action using the target network
            q_val_best_action = self.target_nn.estimate_q(np.array([next_state])).squeeze()
            q_val_best_action = q_val_best_action[best_action]

            # update the value of the action taken
            # in the current state
            if done:
                target[action] = reward
            else:
                target[action] = reward + self.gamma * q_val_best_action
            
            targets.append(target)
        
        return states, targets
    
    def add_observer(self, observer):
        self.observers.append(observer)
