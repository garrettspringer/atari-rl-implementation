import gym
from model import atari_model

env = gym.make('BreakoutDeterministic-v4')
env.render()

# Create model
# How do i make this into an int? env.action_space
model = atari_model(len(env.unwrapped.get_action_meanings()))

for _ in range(1000):
  # Return to the starting frame
  frame = env.reset()
  is_done = False
  while not is_done:
    # Select an action
    #action = model.choose_best_action(frame) 
    # Perform a random action, returns the new frame, reward and whether the game is over
    frame, reward, is_done, _ = env.step(env.action_space.sample())
    env.render()

env.close()