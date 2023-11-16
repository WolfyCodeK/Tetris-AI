import pygame
from controllers.game_controller import GameController
from controllers.window import Window
from game.actions_enum import Actions
from game.game_settings import GameSettings
from enum import IntEnum
from gym.spaces import Discrete
from gym import Env

class ScreenSizes(IntEnum):
        XXSMALL = 6,
        XSMALL = 8,
        SMALL = 10,
        MEDIUM = 12,
        LARGE = 14,
        XLARGE = 16,
        XXLARGE = 18

class TetrisEnv(Env):
    SCREEN_SIZE_TYPE_ERROR = "TetrisEnv.render() -> size must be of types int or str"
    SCREEN_SIZE_STRING_ERROR = f"TetrisEnv.render() -> size must be from list {ScreenSizes._member_names_}"
    
    render_window = False
    
    def __init__(self) -> None:
        self.game = GameController()
        self.action_space = Discrete(len(Actions))
        self.window = None
        
    def _update_window(self):
        if (pygame.event.get(pygame.QUIT)):
            pygame.quit()
            
        self.window.draw()
            
    def step(self, action):     
        self.game.cycle_game_clock()
        self.game.perform_action(action)
        score, done = self.game.run_logic()
        
        if self.window is not None:
            self._update_window()
        
        if done:
            self.game.reset()
        
        return score, done

    def render(self, screen_size: ScreenSizes|int, show_fps: bool):
        # Initial pygame setup
        pygame.display.init()
        pygame.font.init()
        
        # Create window to be rendered
        self.window = Window(self.game)
        
        # Configure window settings
        GameSettings.show_fps_counter = show_fps
        
        if (screen_size in ScreenSizes._value2member_map_):
            GameSettings.set_screen_size(screen_size)
            
        elif (type(screen_size) == int):
            GameSettings.set_screen_size(screen_size)
            
        else:
            raise TypeError(GameSettings.SCREEN_SIZE_TYPE_ERROR)
        
    def seed(self, seed=None):
        GameSettings.seed = seed
        
        return GameSettings.seed   
        
    def reset(self):
        self.game.reset()
        return self.observation_space
    
    def close(self):
        print("Enviroment closed.")