import time
from train_env import TrainTetrisEnv
from itertools import count
from utils.screen_sizes import ScreenSizes

# Code adapted from -> https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html

if __name__ == '__main__':
    env = TrainTetrisEnv()
    env.render(screen_size=ScreenSizes.SMALL, show_fps=True, show_score=True, show_queue=True, playback=False, playback_aps=5)

    while True:
        # Initialize the environment and get its state
        state, info = env.reset()
        
        for t in count():
            action = env.action_space.sample()
            
            start_time = time.time()
            
            observation, reward, terminated, truncated, _ = env.step(action.item())
            
            end_time = time.time()
            
            print(f"Cycle time: {(end_time - start_time) * 1000}ms")
            # print("###################################")
            # print(f"Reward:  {reward}")
            # print(f"Observation:\n  {observation}")
            done = terminated or truncated
            state = observation

            if done:     
                break