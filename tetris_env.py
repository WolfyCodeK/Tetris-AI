import pygame
from controllers.game_controller import GameController
from controllers.window import Window
from game.game_states import GameStates
from game.game_settings import GameSettings
from enum import IntEnum
from gym.spaces import Discrete

class ScreenSizes(IntEnum):
        XXSMALL = 6,
        XSMALL = 8,
        SMALL = 10,
        MEDIUM = 12,
        LARGE = 14,
        XLARGE = 16,
        XXLARGE = 18

class TetrisEnv():
    SCREEN_SIZE_TYPE_ERROR = "TetrisEnv.render() -> size must be of types int or str"
    SCREEN_SIZE_STRING_ERROR = f"TetrisEnv.render() -> size must be from list {ScreenSizes._member_names_}"
    
    render_window = False
    
    def __init__(self) -> None:
        # Create controller for the game
        self.game = GameController()
        self.game_state = GameStates.UPDATE_TIME
        
        self.action_space = Discrete(6)
    
    def render(self, screen_size: ScreenSizes|int, show_fps: bool):
        # Initial pygame setup
        pygame.display.init()
        pygame.font.init()
        
        if (screen_size in ScreenSizes._value2member_map_):
            GameSettings.set_screen_size(screen_size)
            
        elif (type(screen_size) == int):
            GameSettings.set_screen_size(screen_size)
            
        else:
            raise TypeError(GameSettings.SCREEN_SIZE_TYPE_ERROR)
        
        GameSettings.show_fps_counter = show_fps
        self.render_window = True
        self._run_env()
        
    def step(self, action):
        # State machine - Unrendered
        match self.game_state:            
            case GameStates.UPDATE_TIME:
                self.game.update_delta_time()
                self.game.increment_frames_passed()
                self.game.update_fps_counter()
                self.game_state = GameStates.RUN_LOGIC
                
            case GameStates.RUN_LOGIC:
                self.game.run()
                self.game_state = GameStates.UPDATE_TIME

    def close(self):
        print(f"Score: {self.game.score}")
        print("Game Stopped.")
        print(self.game.get_board_state())

    def _run_env(self):
        if self.render_window:        
            # State machine - Rendered
            while running:
                # Check if user has quit the window
                if (pygame.event.get(pygame.QUIT)):
                    running = False
                
                match self.game_state:
                    case GameStates.INIT_STATE:
                        # Run any initialisation code here
                        print("Game Running...")
                        # Init window for rendering enviroment
                        window = Window(self.game)
                        self.game_state = GameStates.UPDATE_TIME
                    
                    case GameStates.UPDATE_TIME:
                        self.game.update_delta_time()
                        self.game.increment_frames_passed()
                        self.game.update_fps_counter()
                        
                        if self.perform_step:
                            self.game_state = GameStates.TAKE_INPUTS
                            self.perform_step = False
                        else:
                            self.game_state = GameStates.DRAW_GAME
                        
                    case GameStates.TAKE_INPUTS:
                        self.game.take_player_inputs(pygame.event.get())
                        self.game_state = GameStates.RUN_LOGIC
                        
                    case GameStates.RUN_LOGIC:
                        self.game.run()
                        self.game_state = GameStates.DRAW_GAME

                    case GameStates.DRAW_GAME:
                        window.draw()
                        self.game_state = GameStates.UPDATE_TIME    
        if self.render_window:
            pygame.quit()