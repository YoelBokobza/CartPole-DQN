import random
from collections import namedtuple,deque
import random

Transition = namedtuple('Transition',('state', 'action', 'next_state', 'reward', 'not_done'))


class ReplayBuffer(object):

    def __init__(self, capacity):
        self.memory = deque([],maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))


    def sample(self, batch_size):
        batch = random.sample(self.memory,batch_size)
        return batch
    
    def sample_last(self):
        batch = self.memory[-1]
        return batch


    def __len__(self):
        return len(self.memory)
