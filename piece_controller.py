import pieces
import numpy as np
import board_utils as bu
from tetramino import Tetramino

class PieceController():
    
    def __init__(self) -> None:
        # Initialise board state to be empty
        self.board_state = np.zeros(shape=(int(bu.BOARD_ROWS + bu.pixel_to_grid_size(bu.DROP_HEIGHT)), bu.BOARD_COLUMNS), dtype=int)
        self.current_piece = self.__create_new_piece()
        
    def draw_board_state(self, board_surface):
        for y in range(len(self.board_state)):
            for x in range(len(self.board_state[0])):
                if (self.board_state[y][x] > 0):
                    bu.draw_rect(x, y, self.current_piece.colour, board_surface)
                    
    def drop_piece(self) -> None:
        if ((self.current_piece.y_pos < bu.BOARD_ROWS) and (not self.__piece_is_blocked(self.board_state, self.current_piece))):
            self.current_piece.set_y_pos(self.current_piece.y_pos + 1)
        else:
            self.__deactivate_piece()
            
    def draw_piece(self, board_surface):
        self.current_piece.draw(board_surface)
            
    def __deactivate_piece(self) -> None:
        self.current_piece.active = False
        self.__place_piece(self.board_state, self.current_piece)
        self.current_piece = self.__create_new_piece()
    
    @staticmethod
    def __create_new_piece() -> Tetramino:
        # Implement 7 bag cycle
        return pieces.ZPiece()
    
    @staticmethod        
    def __place_piece(board_state, piece):
        for i in range(len(piece.occupying_squares)):
            board_state[piece.occupying_squares[i][1]][piece.occupying_squares[i][0]] = 1
            
    @staticmethod
    def __piece_is_blocked(board_state, piece) -> bool:
        blocked = False

        for i in range(len(piece.occupying_squares)):
            if (piece.occupying_squares[i][1] + 1 >= 0):
                piece_pos = board_state[piece.occupying_squares[i][1] + 1][piece.occupying_squares[i][0]]
                
                if (piece_pos > 0):  
                    blocked = True
                    
        return blocked