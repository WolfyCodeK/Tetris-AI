import time
from tetris_env import TetrisEnv
from itertools import count
from utils.screen_sizes import ScreenSizes

# Code adapted from -> https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html

env = TetrisEnv()
env.render(screen_size=ScreenSizes.XXSMALL, show_fps=True)
    
num_episodes = 500_000

for i_episode in range(num_episodes):
    # Initialize the environment and get its state
    state, info = env.reset()
    
    for t in count():
        time.sleep(1)
        action = env.action_space.sample()
        observation, reward, terminated, truncated, _ = env.step(action.item())
        print("###################################")
        print(f"Reward:  {reward}")
        print(f"Observation:\n  {observation}")
        done = terminated or truncated
        state = observation

        if done:     
            break

print('Complete')