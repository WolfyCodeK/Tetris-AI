from tetris_env import TetrisEnv, ScreenSizes

if __name__ == '__main__':
    
    env = TetrisEnv()
    # env.render(screen_size=ScreenSizes.MEDIUM, show_fps=False, show_score=False)
    env.seed(0)

    episode = 1
    highest_score = 0
    
    while highest_score == 0:
        done = False
        
        while not done:
            action = env.action_space.sample()  
            done = env.step(action)
        
        print(f"Episode: {episode}, Score: {env.score}")
        episode += 1
        
        if (env.score > highest_score):
            highest_score = env.score
        else:
            env.reset()
    
    print(f"Highest Score: {highest_score}")
        
    env.close()
    
