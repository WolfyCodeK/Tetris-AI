from tetris_env import TetrisEnv, ScreenSizes

if __name__ == '__main__':
    
    env = TetrisEnv()
    env.render(screen_size=ScreenSizes.XXSMALL, show_fps=False, show_score=False)
    env.seed(0)
    
    episode = 1
    highest_score = 0
    
    while highest_score == 0:
        done = False
        
        while not done:
            action = env.action_space.sample()  
            score, done = env.step(action, actions_per_second=5)
        
        print(f"Episode: {episode}, Score: {score}")
        episode += 1
        
        if (score > highest_score):
            highest_score = score
        else:
            env.reset()
    
    print(f"Highest Score: {highest_score}")
        
    env.close()
    
