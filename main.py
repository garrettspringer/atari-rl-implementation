import gym
from model import atari_model
from memory import RingBuf
from utils import preprocess, to_grayscale
import numpy as np

env = gym.make('BreakoutDeterministic-v4')
env.render()

# Create model
model = atari_model(len(env.unwrapped.get_action_meanings()))

max_num_memories = 1000000
memory_storage = RingBuf(max_num_memories)

warmup_size = 50000
model.warmup(env, warmup_size, memory_storage)

is_done = False
for i in range(1000000):
  frame = preprocess(env.reset())
  model.state_list = [frame]
  is_done = False
  while not is_done:
    for j in range(4):
      # FIXME choose best action
      #action = env.action_space.sample()
      if (len(model.state_list)<4):
        action = env.action_space.sample()
      #else:
        #action = model.choose_best_action([model.state_list[len(model.state_list)-4:len(model.state_list)], np.ones(env.action_space.shape)])
      new_frame, reward, is_done, _ = env.step(action)
      if is_done:
        break
      model.state_list.append(preprocess(new_frame))

    is_done = model.q_iteration(env, model.state_list[len(model.state_list)-4:len(model.state_list)], i, memory_storage)

env.close()