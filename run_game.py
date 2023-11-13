import pygame

from tetris_env import TetrisEnv, ScreenSizes

if __name__ == '__main__':
    new_tetris_env = TetrisEnv()
    
    new_tetris_env.render(screen_size=ScreenSizes.SMALL, show_fps=True)
    new_tetris_env.main()
    pygame.quit()
