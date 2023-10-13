import numpy as np
import board_utils as bu
import tetramino_features as tf
import random

class PieceController():

    NUM_OF_PIECES = 5
    PIECE_NUMBERS = list(range(0, NUM_OF_PIECES))
    bag_counter = 0
    
    def __init__(self) -> None:
        # Initialise board state to be empty
        self.board_state = np.full(shape=(int(bu.BOARD_ROWS + bu.pixel_to_grid_size(bu.DROP_HEIGHT)), bu.BOARD_COLUMNS), fill_value=tf.EMPTY_PIECE_PID)
        self.new_piece()
        
    def draw_deactivated_pieces(self, board_surface):
        for y in range(len(self.board_state)):
            for x in range(len(self.board_state[0])):
                if (self.board_state[y][x] != tf.EMPTY_PIECE_PID):
                    bu.draw_rect(x, y, tf.COLOUR_PID_DICT[self.board_state[y][x]], board_surface)
            
    def draw_current_piece(self, board_surface):
        self.current_piece.draw(board_surface)
        
    def gravity_drop_piece(self) -> bool:
        """Attempts to drop a tetramino piece down by one row.

        Returns:
            bool: True if the piece was successfully dropped down one row
        """
        if ((self.current_piece.y_pos < bu.BOARD_ROWS) and (not self.__piece_is_vertically_blocked(self.board_state, self.current_piece))):
            self.current_piece.set_y_pos(self.current_piece.y_pos + 1)
            
            return True
        else:
            return False
        
    def shift_piece_horizontally(self, x):
        if (not self.__piece_is_horizontally_blocked(self.board_state, self.current_piece, x)):
            self.current_piece.set_x_pos(self.current_piece.x_pos + x)
            
    def deactivate_piece(self) -> None:
        self.current_piece.active = False
        self.__place_piece(self.board_state, self.current_piece)
    
    def new_piece(self) -> None:
        if (self.bag_counter <= 0):
            random.shuffle(self.PIECE_NUMBERS)
            self.bag_counter = self.NUM_OF_PIECES
            
        self.bag_counter -= 1
        
        piece_num = self.PIECE_NUMBERS[self.bag_counter]

        self.current_piece = tf.PIECE_CLASS_LIST[piece_num]() 
    
    @staticmethod        
    def __place_piece(board_state, piece):
        for i in range(len(piece.occupying_squares)):
            board_state[piece.occupying_squares[i][1]][piece.occupying_squares[i][0]] = piece.pid
            
    @staticmethod
    def __piece_is_vertically_blocked(board_state, piece) -> bool:
        blocked = False

        for i in range(len(piece.occupying_squares)):
            # Check if piece is in the board
            if (piece.occupying_squares[i][1] + 1 >= 0):
                pos_state = board_state[piece.occupying_squares[i][1] + 1][piece.occupying_squares[i][0]]
                
                # If there is a piece there then the position is blocked
                if (pos_state != tf.EMPTY_PIECE_PID):  
                    blocked = True
                    
        return blocked
    
    @staticmethod
    def __piece_is_horizontally_blocked(board_state, piece, x) -> bool:
        blocked = False

        for i in range(len(piece.occupying_squares)):
            piece_pos = piece.occupying_squares[i][0] + x
            
            if (x > 0):
                if (piece_pos + x <= bu.BOARD_COLUMNS):
                    if (board_state[piece.occupying_squares[i][1]][piece.occupying_squares[i][0] + x] != tf.EMPTY_PIECE_PID):
                        blocked = True
                else:
                    blocked = True
            
            if (x < 0):
                if (piece_pos + x >= -1):
                    if (board_state[piece.occupying_squares[i][1]][piece.occupying_squares[i][0] + x] != tf.EMPTY_PIECE_PID):
                        blocked = True
                else:
                    blocked = True          
            
        return blocked