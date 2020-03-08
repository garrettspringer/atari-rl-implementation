import gym
from model import atari_model
from memory import RingBuf

env = gym.make('BreakoutDeterministic-v4')
env.render()

# Create model
# How do i make this into an int? env.action_space
model = atari_model(len(env.unwrapped.get_action_meanings()))

for i in range(1000):
  # Return to the starting frame
  frame = env.reset()
  is_done = False
  while not is_done:
    model.q_iteration(env, frame, i, RingBuf)

env.close()