import pygame
from board.board import Board
from controllers.logic_controller import LogicController
from pieces.piece_controller import PieceController
from controllers.window_controller import WindowController
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
        # Create game board
        board = Board()

        # Create controllers for the game
        p_controller = PieceController(board)
        l_controller = LogicController(p_controller)
        
        # Init window controller for rendering window if enviroment needs rendering
        if self.render_window:
            w_controller = WindowController(l_controller)

        game_state = GameStates.INIT_STATE
        
        running = True
        
        while running:
            # Check if user has quit the window
            if self.render_window:
                if (pygame.event.get(pygame.QUIT)):
                    running = False
            
            match game_state:
                case GameStates.INIT_STATE:
                    print("Game Running...")
                    # Run any initialisation code here
                    game_state = GameStates.UPDATE_TIME
                
                case GameStates.UPDATE_TIME:
                    l_controller.update_delta_time()
                    l_controller.increment_frames_passed()
                    l_controller.update_fps_counter()
                    
                    if self.render_window:
                        game_state = GameStates.TAKE_INPUTS
                    else:
                        game_state = GameStates.RUN_LOGIC
                    
                case GameStates.TAKE_INPUTS:
                    l_controller.take_player_inputs(pygame.event.get())
                    game_state = GameStates.RUN_LOGIC
                    
                case GameStates.RUN_LOGIC:
                    l_controller.run()

                    if self.render_window:
                        game_state = GameStates.DRAW_GAME
                    else:
                        game_state = GameStates.UPDATE_TIME
                    
                case GameStates.DRAW_GAME:
                    w_controller.draw()
                    game_state = GameStates.UPDATE_TIME
        
        print("Game Stopped.")