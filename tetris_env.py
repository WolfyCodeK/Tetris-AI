import pygame
from controllers.game_controller import GameController
from controllers.window import Window
from game.game_states import GameStates
from game.game_settings import GameSettings
from enum import IntEnum

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
        pass
    
    def render(self, screen_size: ScreenSizes|int, show_fps: bool):
        if (screen_size in ScreenSizes._value2member_map_):
            GameSettings.set_screen_size(screen_size)
            
        elif (type(screen_size) == int):
            GameSettings.set_screen_size(screen_size)
            
        else:
            raise TypeError(GameSettings.SCREEN_SIZE_TYPE_ERROR)
        
        GameSettings.show_fps_counter = show_fps
        self.render_window = True

    def main(self):
        # Create controller for the game
        game = GameController()
        game_state = GameStates.INIT_STATE
        
        running = True
        
        if self.render_window:
            # Initial pygame setup
            pygame.display.init()
            pygame.font.init()
                
            # State machine - Rendered
            while running:
                # Check if user has quit the window
                if (pygame.event.get(pygame.QUIT)):
                    running = False
                
                match game_state:
                    case GameStates.INIT_STATE:
                        # Run any initialisation code here
                        print("Game Running...")
                        # Init window for rendering enviroment
                        window = Window(game)
                        game_state = GameStates.UPDATE_TIME
                    
                    case GameStates.UPDATE_TIME:
                        game.update_delta_time()
                        game.increment_frames_passed()
                        game.update_fps_counter()
                        game_state = GameStates.TAKE_INPUTS
                        
                    case GameStates.TAKE_INPUTS:
                        game.take_player_inputs(pygame.event.get())
                        game_state = GameStates.RUN_LOGIC
                        
                    case GameStates.RUN_LOGIC:
                        game.run()
                        game_state = GameStates.DRAW_GAME

                    case GameStates.DRAW_GAME:
                        window.draw()
                        game_state = GameStates.UPDATE_TIME
        else:
            # State machine - Unrendered
            for i in range(10000):
                match game_state:
                    case GameStates.INIT_STATE:
                        # Run any initialisation code here
                        print("Game Running...")
                        game_state = GameStates.UPDATE_TIME
                    
                    case GameStates.UPDATE_TIME:
                        game.update_delta_time()
                        game.increment_frames_passed()
                        game.update_fps_counter()
                        game_state = GameStates.RUN_LOGIC
                        
                    case GameStates.RUN_LOGIC:
                        game.run()
                        game_state = GameStates.UPDATE_TIME
        
        print(f"Score: {game.score}")
        print("Game Stopped.")
        pygame.quit()