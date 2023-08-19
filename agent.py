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