import gymnasium as gym
from agent import Agent, Transition
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
            print(f"Action at step {t}: {select_act.item()} | Raw format: {select_act}")

            # Do action and receive feedback from the environment
            observation, reward, terminated, truncated, _ = env.step(select_act.item())
            is_done = terminated or truncated

            if terminated:
                next_state = None
            else:
                next_state = torch.tensor(observation, dtype = torch.float32,
                                          device = device).unsqueeze(0)
                
            mb_agen.memory.push(state, select_act, next_state, reward)
            state = next_state

            # Check the output variable of ReplayMemory

            if t > 10 or is_done:
                break

            #@TODO: Implement training model
            mb_agen.optimize_model()

            target_net_state_dict = mb_agen.target_net.state_dict()
            policy_net_state_dict = mb_agen.policy_net.state_dict()

            for key in policy_net_state_dict:
                target_net_state_dict[key] = 
        
    sample_observation = mb_agen.memory.sample(5)
    m = Transition(*zip(*sample_observation))

    state_batch = torch.cat(m.state)
    action_batch = torch.cat(m.action)

    with torch.no_grad():
        result = mb_agen.policy_net(state_batch).gather(1, action_batch)

    print(f"The value of result: {result.shape}")
    print(f"The value of action_batch: {action_batch}")