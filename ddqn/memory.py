from collections import deque
import random

class ReplayMemory(object):
    def __init__(self, buffer_size):
        """
        Initializes a replay buffer.

        Args:
            buffer_size
        """
        self.buffer = deque([], maxlen=buffer_size)
    
    def store(self, experiences):
        """
        Store a list of experiences into the replay buffer.

        Args:
            experiences
        """
        self.buffer.extend(experiences)
    
    def sample(self, num_samples):
        """
        Samples past experiences from the memory buffer.
        
        Args:
            num_samples
        """
        num_samples = min(num_samples, len(self.buffer))
        random_sample = random.sample(self.buffer, num_samples)

        return random_sample