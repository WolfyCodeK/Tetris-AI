import pieces
import numpy as np
import board_utils as bu
from tetramino import Tetramino
import tetramino_features as tf
import random

class PieceController():

    NUM_OF_PIECES = 4

    drop_speed = 4
    drop_time = 1 / drop_speed
    
    piece_numbers = list(range(1, NUM_OF_PIECES + 1))
    bag_counter = 0
    
    def __init__(self) -> None:
        # Initialise board state to be empty
        self.board_state = np.zeros(shape=(int(bu.BOARD_ROWS + bu.pixel_to_grid_size(bu.DROP_HEIGHT)), bu.BOARD_COLUMNS), dtype=int)
        self.current_piece = self.__create_new_piece()
        
    def draw_board_state(self, board_surface):
        for y in range(len(self.board_state)):
            for x in range(len(self.board_state[0])):
                if (self.board_state[y][x] == tf.Z_PIECE_PID):
                    bu.draw_rect(x, y, tf.Z_PIECE_COLOUR, board_surface)
                if (self.board_state[y][x] == tf.L_PIECE_PID):
                    bu.draw_rect(x, y, tf.L_PIECE_COLOUR, board_surface)
                if (self.board_state[y][x] == tf.S_PIECE_PID):
                    bu.draw_rect(x, y, tf.S_PIECE_COLOUR, board_surface)
                if (self.board_state[y][x] == tf.J_PIECE_PID):
                    bu.draw_rect(x, y, tf.J_PIECE_COLOUR, board_surface)
                    
    def drop_piece(self) -> None:
        if ((self.current_piece.y_pos < bu.BOARD_ROWS) and (not self.__piece_is_vertically_blocked(self.board_state, self.current_piece))):
            self.current_piece.set_y_pos(self.current_piece.y_pos + 1)
        else:
            self.__deactivate_piece()
            
    def draw_piece(self, board_surface):
        self.current_piece.draw(board_surface)
        
    def shift_piece_by_amount(self, x):
        if (not self.__piece_is_horizontally_blocked(self.board_state, self.current_piece, x)):
            self.current_piece.set_x_pos(self.current_piece.x_pos + x)
            
    def __deactivate_piece(self) -> None:
        self.current_piece.active = False
        self.__place_piece(self.board_state, self.current_piece)
        self.current_piece = self.__create_new_piece()
    
    def __create_new_piece(self) -> Tetramino:
        if (self.bag_counter <= 0):
            random.shuffle(self.piece_numbers)
            self.bag_counter = self.NUM_OF_PIECES
            
        self.bag_counter -= 1
        
        piece_num = self.piece_numbers[self.bag_counter]
        
        if (piece_num == tf.Z_PIECE_PID):
            return pieces.ZPiece()
        if (piece_num == tf.L_PIECE_PID):
            return pieces.LPiece()
        if (piece_num == tf.S_PIECE_PID):
            return pieces.SPiece()
        if (piece_num == tf.J_PIECE_PID):
            return pieces.JPiece()
        if (piece_num == 5):
            return pieces.ZPiece()
        if (piece_num == 6):
            return pieces.LPiece()
        if (piece_num == 7):
            return pieces.ZPiece()
    
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
                if (pos_state > 0):  
                    blocked = True
                    
        return blocked
    
    @staticmethod
    def __piece_is_horizontally_blocked(board_state, piece, x) -> bool:
        blocked = False

        for i in range(len(piece.occupying_squares)):
            piece_pos = piece.occupying_squares[i][0] + x
            
            if (x > 0):
                if (piece_pos + x <= bu.BOARD_COLUMNS):
                    if (board_state[piece.occupying_squares[i][1]][piece.occupying_squares[i][0] + x] > 0):
                        blocked = True
                else:
                    blocked = True
            
            if (x < 0):
                if (piece_pos + x >= -1):
                    if (board_state[piece.occupying_squares[i][1]][piece.occupying_squares[i][0] + x] > 0):
                        blocked = True
                else:
                    blocked = True          
            
        return blocked