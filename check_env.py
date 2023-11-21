from stable_baselines3.common.env_checker import check_env
from tetris_env import TetrisEnv, ScreenSizes

if __name__ == '__main__':
    env = TetrisEnv()
    env.render(screen_size=ScreenSizes.MEDIUM, show_fps=True, show_score=True)
    env.seed(0)
    
    check_env(env, skip_render_check=True)

    episode = 1
    highest_score = 0
    
    while highest_score == 0:
        done = False
        obs = env.reset()
        
        while not done:
            action = env.action_space.sample()  
            obs, reward, done, info = env.step(action)
        
        print(f"Episode: {episode}, Score: {env.game_score}")
        episode += 1
        
        if (env.game_score > highest_score):
            highest_score = env.game_score
    
    print(f"Highest Score: {highest_score}")
        
    env.close()