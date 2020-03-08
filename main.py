import gym
from model import atari_model
from memory import RingBuf
from utils import preprocess, to_grayscale

env = gym.make('BreakoutDeterministic-v4')
env.render()

# Create model
model = atari_model(len(env.unwrapped.get_action_meanings()))

max_num_memories = 1000000
memory_storage = RingBuf(max_num_memories)

for i in range(1000):
  # Return to the starting frame
  frame = env.reset()
  frame = preprocess(frame)
  is_done = False
  while not is_done:
    model.q_iteration(env, frame, i, memory_storage)

env.close()