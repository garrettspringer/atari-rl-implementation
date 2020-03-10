import gym
from model import atari_model
from memory import RingBuf
from utils import preprocess, to_grayscale, transform_reward
import numpy as np

env = gym.make('BreakoutDeterministic-v4')
env.render()

# Create model
model = atari_model(env.unwrapped.get_action_meanings(), env.action_space.n)

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
    for j in range(3):
      action = env.action_space.sample()
      new_frame, reward, is_done, _ = env.step(action)
      model.episode_reward += transform_reward(reward)
      model.state_list.append(preprocess(new_frame))
      env.render()

    action = model.choose_best_action([model.previous_state()])
    new_frame, reward, is_done, _ = env.step(action)
    model.episode_reward += transform_reward(reward)

    env.render()

    if is_done:
      model.reset_episode_scores()
      break
    
    model.state_list.append(preprocess(new_frame))
    is_done = model.q_iteration(env, model.previous_state(), i, memory_storage)

env.close()