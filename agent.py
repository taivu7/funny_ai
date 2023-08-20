from collections import namedtuple, deque
import random
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.optim as optim
import math
from gymnasium.core import Env

Transition = namedtuple('Transition', 
                        ('state', 'action', 'next_state', 'reward'))

BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000
TAU = 0.005
LR = 1e-4

class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)
    
    def push(self, *args):
        """ Add transition into memory
        """
        trans = Transition(*args)
        self.memory.append(trans)
    
    def sample(self, sample_size: int):
        """ Get randomly transitions from memory

        Args:
            sample_size (int): _description_
        """

        sample_ls = random.sample(self.memory, sample_size)
        return sample_ls

    def __len__(self):
        """ The size of current memory
        """
        return len(self.memory)

class DQN(nn.Module):

    def __init__(self, n_observations, n_actions) -> None:
        super().__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)
    
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        return x

class Agent:

    def __init__(self, n_observations, n_actions, mem_capacity, device):
        self.n_observations = n_observations
        self.n_actions = n_actions
        self.memory = ReplayMemory(mem_capacity)
        self.device = device
        self.policy_net = DQN(n_observations, n_actions).to(device)
        self.target_net = DQN(n_observations, n_actions).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.step_done = 0

        # optimizer
        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=LR, amsgrad=True)

    def env_interact(self, state, env: Env):
        """ Select action based on e-greedy
        """

        # generate a random value
        random_value = random.random()
        eps_value = EPS_END + (EPS_START - EPS_END) * \
            math.exp(-1.0 * self.step_done / EPS_DECAY)
        
        self.step_done += 1

        # with torch.no_grad():
        #     tmp = self.policy_net(state)
        #     print(f"Output of network: {tmp}")
        #     print(f"Max of network: {tmp.max(1)[1].view(1, 1)}")

        if random_value > eps_value:
            with torch.no_grad():
                return self.policy_net(state).max(1)[1].view(1, 1)
        else:
            return torch.tensor([[env.action_space.sample()]], dtype=torch.long,
                                device = self.device)
    
    def optimize_model(self):

        if len(self.memory) < BATCH_SIZE:
            return
        
        sample = self.memory.sample(BATCH_SIZE)
        batch = Transition(*zip(*sample))

        # Aggregate data in each field into batch variables
        non_final_mask = torch.tensor(
            tuple(map(lambda s: s is not None, batch.next_state)),
            dtype = torch.bool,
            device = self.device
        )

        non_final_next_states = torch.cat(
            [s for s in batch.next_state if s is not None],
            dim = 0
        )

        state_batch = torch.cat(batch.action)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        state_action_value = self.policy_net(state_batch).gather(1, action_batch) # 2-d array whose shape is (batch_size, 1)

        next_state_values = torch.zeros(BATCH_SIZE, device = self.device)

        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0]
            # The shape of next_state_values is a 1-d array whose size (BATCH_SIZE)
        
        # Compute the expected Q values
        expected_state_action_val = (next_state_values * GAMMA) + reward_batch

        # Compute loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_value, expected_state_action_val.unsqueeze(1))

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()

        #In-place gradient clipping and update weight
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()