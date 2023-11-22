from tetris_env import TetrisEnv, ScreenSizes
from stable_baselines3 import PPO, A2C
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
    env.render(screen_size=ScreenSizes.MEDIUM, show_fps=True, show_score=True)
    env.reset()

    model = A2C('MlpPolicy', env, verbose=1, tensorboard_log=logs_directory)

    STEPS = 12000
    count = 0
    
    while True:
        count += 1
        print(env.reward)
        model.learn(total_timesteps=STEPS, reset_num_timesteps=False, tb_log_name=f"A2C")
        model.save(f"{models_directory}/{STEPS*count}")