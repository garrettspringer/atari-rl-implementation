import gym
from model import atari_model

env = gym.make('BreakoutDeterministic-v4')
env.render()

for _ in range(10000):
  # Return to the starting frame
  frame = env.reset()
  is_done = False
  while not is_done:
    # Number of actions in the game
    n_actions = env.action_space
    # Create model
    model = atari_model(n_actions)
    # Select an action
    action = model.choose_best_action(frame)   

    # Perform a random action, returns the new frame, reward and whether the game is over
    frame, reward, is_done, _ = env.step(env.action_space.sample())
    env.render()

env.close()