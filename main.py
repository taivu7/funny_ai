import gymnasium as gym
from agent import DQN
from itertools import count

if __name__ == "__main__":

    env = gym.make("CartPole-v1")
    n_actions = env.action_space.n
    print(f'Total actions: {n_actions}')
    
    state, info = env.reset()

    # Interact to the environment
    num_episode = 1

    for _ in range(num_episode):
        state, info = env.reset()

        for t in count():
            action = env.action_space.sample()
            observation, reward, terminated, truncated, _ = env.step(action)
            print(f"{t}: {reward}")

            done = terminated or truncated

            if done:
                break

    print(f'The shape of state: {state.shape}')