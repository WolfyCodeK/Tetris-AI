from tetris_env import TetrisEnv, ScreenSizes
from gym import make

if __name__ == '__main__':
    testEnv = make("Blackjack-v1")
    
    env = TetrisEnv()
    action = env.action_space.sample()
    # env.render(ScreenSizes.LARGE, True)
    for i in range(100000):
        env.step(action)
    env.close()
