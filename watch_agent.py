from tetris_env import TetrisEnv, ScreenSizes
from stable_baselines3 import PPO, A2C
import os
from datetime import datetime

if __name__ == '__main__':
    env = TetrisEnv()
    env.render(screen_size=ScreenSizes.MEDIUM, show_fps=True, show_score=True)
    env.seed()

    model = A2C.load("models/2023-11-22---02-37-27/384000.zip", env=env)
    
    episode = 1
    highest_score = 0
    
    while highest_score == 0:
        done = False
        obs = env.reset()
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
        
        print(f"Episode: {episode}, Score: {env.game_score}, Reward: {env.reward}")
        episode += 1
        
        if (env.game_score > highest_score):
            highest_score = env.game_score
    
    print(f"Highest Score: {highest_score}")
        
    env.close()