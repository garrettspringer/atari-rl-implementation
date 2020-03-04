import gym

env = gym.make('BreakoutDeterministic-v4')
# returns the starting frame
frame = env.reset()
env.render()

is_done = False
while not is_done:
  # Perform a random action, returns the new frame, reward and whether the game is over
  frame, reward, is_done, _ = env.step(env.action_space.sample())
  env.render()