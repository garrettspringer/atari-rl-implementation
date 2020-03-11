import gym
from model import atari_model
from memory import RingBuf
from utils import preprocess, to_grayscale, transform_reward
import numpy as np

if __name__ == "__main__":
    env = gym.make('BreakoutDeterministic-v4')
    env.render()

    # Create model
    model = atari_model(env.unwrapped.get_action_meanings(), env.action_space.n)

    max_num_memories = 1000000
    memory_storage = RingBuf(max_num_memories)

    warmup_size = 25
    model.warmup(env, warmup_size, memory_storage)

    is_done = False
    iteration = 0
    while True:
        frame = preprocess(env.reset())
        model.state_list = [frame]
        
        iteration = model.q_iteration(env, iteration, memory_storage)
        iteration+=1

    model._save_weights()
    env.close()