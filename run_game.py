import time
from tetris_env import TetrisEnv, ScreenSizes

if __name__ == '__main__':
    env = TetrisEnv()
    env.render(ScreenSizes.LARGE, True)
    env.seed(1)
    episodes = 10
    episode = 1
    highest_score = 0
    
    while highest_score == 0:
        done = False
        
        while not done:
            action = env.action_space.sample()
            score, done = env.step(action)
            time.sleep(2)
        
        print(f"Episode: {episode}, Score: {score}")
        episode += 1
        
        if (score > highest_score):
            highest_score = score
        else:
            env.reset()
    
    print(f"Highest Score: {highest_score}")
        
    env.close()
    
