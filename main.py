import gymnasium as gym
from agent import Agent
from itertools import count
import torch

if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = gym.make("CartPole-v1")   
    n_actions = env.action_space.n
    print(f'Total actions: {n_actions}')

    state, info = env.reset()
    n_observations = len(state)

    # Interact to the environment
    num_episode = 1

    mb_agen = Agent(n_observations = n_observations,
                    n_actions = n_actions,
                    mem_capacity = 1000, device = device)

    for _ in range(num_episode):
        state, info = env.reset()
        state = torch.tensor(state, dtype=torch.float32, device = device).unsqueeze(0)
        for t in count():
            select_act = mb_agen.env_interact(state, env)
            print(f"Action at step {t}: {select_act.item()}")

            # Do action and receive feedback from the environment
            observation, reward, terminated, truncated, _ = env.step(select_act.item())
            is_done = terminated or truncated

            if terminated:
                next_state = None
            else:
                next_state = torch.tensor(observation, dtype = torch.float32,
                                          device = device).unsqueeze(0)
            state = next_state

            if t > 5 or is_done:
                break


