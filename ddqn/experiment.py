import gym

class Experiment(object):
    def __init__(self, key):
        self.key = key
        self.env = gym.make(key)
    
    def run(self):
        pass

    def run_episode(self):
        pass


