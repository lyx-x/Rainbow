import random
from collections import deque, namedtuple


HalfTransition = namedtuple('Transition', ('state', 'action', 'reward', 'done'))
Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state'))


# TODO: Discretise memory?
class ReplayMemory():
  def __init__(self, capacity):
    self.capacity = capacity
    self.memory = deque([], maxlen=capacity)

  def append(self, *args):
    self.memory.append(HalfTransition(*args))

  def sample(self, batch_size):
    transitions = []
    for _ in range(batch_size):  # TODO: Get efficient indexing for retrieving valid transitions
      i = random.randrange(len(self) - 1)
      s, a, r, d = self.memory[i]
      ns, _, _, _ = self.memory[i + 1]  # TODO: Make sure ns is None for terminal transitions
      transitions.append(Transition(s, a, r, ns))
    transitions = Transition(*zip(*transitions))  # Transpose the batch
    return transitions

  def __len__(self):
    return len(self.memory)

  def __getitem__(self, key):
    return self.memory[key]

# TODO: Prioritised experience replay memory
