from collections import namedtuple, deque
import random
import torch.nn as nn
import torch.nn.functional as F
import torch

Transition = namedtuple('Transition', 
                        ('state', 'action', 'next_state', 'reward'))

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

def select_action(state):

    global steps_done
    sample = random.random()
    return torch.tensor()
