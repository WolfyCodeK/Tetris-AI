import time
from environments.train_env import TrainTetrisEnv
from itertools import count
from utils.screen_sizes import ScreenSizes

# Code adapted from -> https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html

# This file is used for testing the training enviroment with random actions

if __name__ == '__main__':
    env = TrainTetrisEnv()
    env.render(screen_size=ScreenSizes.SMALL, show_fps=True, show_score=True, show_queue=True, playback=True, playback_aps=5)

    while True:
        # Initialize the environment and get its state
        state, info = env.reset()
        
        cycle_times = []
        
        for t in count():
            action = env.action_space.sample()
            
            start_time = time.time()
            
            observation, reward, terminated, truncated, _ = env.step(action.item())
            
            end_time = time.time()
            
            cycle_times.append((end_time - start_time) * 1000)
            print("###################################")
            print(f"Reward:  {reward}")
            print(f"Observation:\n  {observation}")
            done = terminated or truncated
            state = observation

            if done:    
                print(f"Average cycle time: {(sum(cycle_times) / len(cycle_times))}ms") 
                break