from tetris_env import TetrisEnv, ScreenSizes

if __name__ == '__main__':
    env = TetrisEnv()
    env.render(ScreenSizes.LARGE, True)
    env.main()
