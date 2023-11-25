from tetris_env import TetrisEnv, ScreenSizes
from stable_baselines3 import PPO, A2C
import os
from datetime import datetime

if __name__ == '__main__':
    env = TetrisEnv()
    env.render(screen_size=ScreenSizes.MEDIUM, show_fps=True, show_score=True)
    env.seed()

    model = PPO.load("models/2023-11-24---21-28-32/7380000.zip", env=env)
    
    episode = 1
    highest_score = 0
    
    while True:
        done = False
        obs = env.reset()
        
        while not done:
            action, _ = model.predict(obs)
            obs, reward, done, info = env.step(action)
        
        print(f"Episode: {episode}, Score: {env.game_score}, Reward: {env.reward}")
        episode += 1