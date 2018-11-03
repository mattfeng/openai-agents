import tensorflow as tf

GAMMA = 0.99
NUM_EPISODES = 5000
MAX_STEPS = 999

def discounted_returns(rewards, normalize=True):
    """
    Args:
        rewards (np.array): Array of rewards at each timestep.
        normalize (bool): Should the returns be normalized?
    Returns:
        Array of discounted returns `G` (sum over discounted rewards).
    """
    discounted_g = np.zeros_like(rewards)
    cumsum = 0
    for t in range(rewards.size - 1, -1, -1):
        cumsum = cumsum * GAMMA + rewards[t]
        discounted_g[t] = cumsum
    
    mean = np.mean(discounted_g)
    stdev = np.stdev(discounted_g)

    return (discounted_g - mean) / stdev


def test():
    pass

def train():


def main():
    for i in 
    

if __name__ == "__main__":
    main()