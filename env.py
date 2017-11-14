import logging
from collections import deque
from skimage import color, transform
import gym
import torch

# Disable gym logging
logging.disable(logging.INFO)


class Env():
  def __init__(self, args):
    super().__init__()
    self.dtype = torch.cuda.FloatTensor if args.cuda else torch.FloatTensor

    self.env = gym.make(args.game + 'Deterministic-v4')
    self.window = args.history_length  # Number of frames to concatenate
    self.buffer = deque([], maxlen=args.history_length)
    self.t = 0  # Internal step counter
    self.T = args.max_episode_length
    self.training = True  # Consistent with model training mode
    self.lives = 0  # Life counter (used in DeepMind training)

  # TODO: Check these - quite probably result in states that are not properly discretised to [0, 255]
  def _state_to_tensor(self, state):
    gray_img = color.rgb2gray(state)  # TODO: Check image conversion doesn't cause problems
    downsized_img = transform.resize(gray_img, (84, 84), mode='constant')  # TODO: Check resizing doesn't cause problems
    state = torch.from_numpy(downsized_img).type(self.dtype)  # 2D image tensor
    return state

  def _reset_buffer(self):
    for t in range(self.window):
      self.buffer.append(self.dtype(84, 84).zero_())

  def reset(self):
    # Reset internals
    self.t = 0
    self._reset_buffer()
    self.lives = self.env.env.ale.lives()
    # Process and return initial state
    observation = self.env.reset()
    # TODO: 30 random no-op starts?
    observation = self._state_to_tensor(observation)
    self.buffer.append(observation)
    return torch.stack(self.buffer, 0)

  def step(self, action):
    # Process state
    observation, reward, done, _ = self.env.step(action)
    observation = self._state_to_tensor(observation)
    self.buffer.append(observation)
    # Detect loss of life as terminal in training mode
    if self.training:
      lives = self.env.env.ale.lives()
      if lives < self.lives:
        done = True
      else:
        self.lives = lives
    # Time out episode if necessary
    self.t += 1
    if self.t == self.T:
      done = True
    # Return state, reward, done
    return torch.stack(self.buffer, 0), reward, done

  # Uses loss of life as terminal signal
  def train(self):
    self.training = True

  # Uses standard terminal signal
  def eval(self):
    self.training = False

  def action_space(self):
    return self.env.action_space.n

  def seed(self, seed):
    self.env.seed(seed)

  def render(self):
    self.env.render()

  def close(self):
    self.env.close()
