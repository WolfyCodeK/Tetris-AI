from tetris_env import TetrisEnv, ScreenSizes
from stable_baselines3 import PPO, A2C, DQN
import os
from datetime import datetime

if __name__ == '__main__':
    logs_directory = f"logs/{datetime.now().strftime('%Y-%m-%d---%H-%M-%S')}/"
    models_directory = f"models/{datetime.now().strftime('%Y-%m-%d---%H-%M-%S')}/"

    if not os.path.exists(logs_directory):
        os.makedirs(logs_directory)

    if not os.path.exists(models_directory):
        os.makedirs(models_directory)
    
    env = TetrisEnv()
    env.render(screen_size=ScreenSizes.LARGE, show_fps=False, show_score=False)
    env.reset()
    # env.seed(0)

    model = PPO('MlpPolicy', env, verbose=0, tensorboard_log=logs_directory, learning_rate=0.0003)
    # model = PPO.load("models/2023-11-24---15-35-56/960000.zip", env=env)
    model.verbose = 0

    STEPS = 20000
    count = 0
    
    print("Training agent...")
    
    while True:
        count += 1
        model.learn(total_timesteps=STEPS, reset_num_timesteps=False, tb_log_name=f"DQN")
        model.save(f"{models_directory}/{STEPS*count}")