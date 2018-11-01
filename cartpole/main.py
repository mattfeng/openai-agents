import tensorflow as tf

GAMMA = 0.99
NUM_EPISODES = 5000
MAX_STEPS = 999

def discounted_returns(rewards):
    """
    Args:
        rewards (np.array): Array of rewards at each timestep.
    Returns:
        Array of discounted returns `G` (sum of rewards).
    """
    discounted_g = np.zeros_like(rewards)
    cumsum = 0
    for t in range(rewards.size - 1, -1, -1):
        cumsum = cumsum * GAMMA + rewards[t]
        discounted_g[t] = cumsum
    return dicounted_g


def test():
    pass

def train():


def main():
    for i in 
    

if __name__ == "__main__":
    main()