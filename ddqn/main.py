from agent import DDQNAgent
from itertools import count
from collections import deque
import gym

NUM_EPISODES = 10000

def main():
    env = gym.make("CartPole-v0")
    agent = DDQNAgent(env)
    returns = deque(maxlen=100)

    for episode in range(1, NUM_EPISODES + 1):
        state = env.reset()
        done = False
        return_ = 0

        for j in count(1):
            action = agent.act(state)
            state_, reward, done, _ = env.step(action)
            return_ += reward
            loss = agent.learn((state, action, reward, state_, done))
            # env.render()

            if done:
                returns.append(return_)
                avg_return = sum(returns) / len(returns)
                print(f"[ep{episode}] episode len:{j},"
                      f"avg return: {avg_return:.6f}, "
                      f"current ret: {return_:.6f}")
                break

            # transition
            state = state_



if __name__ == "__main__":
    main()