import time
from env import TetrisEnv
from itertools import count
from utils.screen_sizes import ScreenSizes

# Code adapted from -> https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html

if __name__ == '__main__':
    env = TetrisEnv()
    env.render(screen_size=ScreenSizes.XXSMALL, show_fps=True, show_score=True, show_queue=True)
        
    num_episodes = 500_000

    for i_episode in range(num_episodes):
        # Initialize the environment and get its state
        state, info = env.reset()
        
        for t in count():
            action = env.action_space.sample()
            observation, reward, terminated, truncated, _ = env.step(action.item())
            print("###################################")
            print(f"Reward:  {reward}")
            print(f"Observation:\n  {observation}")
            done = terminated or truncated
            state = observation

            if done:     
                break
            
            time.sleep(0.1)

    print('Complete')