from stable_baselines3.common.env_checker import check_env
from gym_tetris_env import TetrisEnv, ScreenSizes

if __name__ == '__main__':
    env = TetrisEnv()
    
    check_env(env, skip_render_check=True)
    
    env.render(screen_size=ScreenSizes.MEDIUM, show_fps=True, show_score=True)

    episode = 1
    highest_score = 0
    
    while highest_score == 0:
        done = False
        obs = env.reset()
        
        while not done:
            action = env.action_space.sample()  
            obs, reward, done, info = env.step(action)
        
        print(f"Episode: {episode}, Score: {env.game_score}, Reward: {env.reward}")
        episode += 1
        
        if (env.game_score > highest_score):
            highest_score = env.game_score
    
    print(f"Highest Score: {highest_score}")
        
    env.close()