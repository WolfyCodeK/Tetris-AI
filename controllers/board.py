import numpy as np
import utils.board_constants as bc

import utils.window_utils as win_utils
from pieces.i_piece import IPiece
from pieces.o_piece import OPiece
from pieces.three_wide_pieces import (JPiece, LPiece, SPiece, TPiece, ZPiece)

class Board():
    # All the available pieces to the piece controller
    PIECE_LIST = [ZPiece, SPiece, JPiece, LPiece, TPiece, IPiece, OPiece]
    
    EMPTY_PIECE_ID = 0
    
    # Dict of (ID, colour) pairs for all pieces
    PIECE_COLOUR_DICT = {}
    for i in range(len(PIECE_LIST)):
        PIECE_COLOUR_DICT[PIECE_LIST[i].ID] = PIECE_LIST[i].COLOUR
        
    def __init__(self) -> None:
        self.INITIAL_BOARD_STATE = self._init_board_state()
        self.occupied_spaces = 0
        self.reset_board_state()
    
    def draw(self, surface):
        for y in bc.BOARD_HEIGHT_RANGE_INCR:
            for x in range(bc.BOARD_WIDTH):
                id = self.board_state[y][x]
                
                if (id in Board.PIECE_COLOUR_DICT.keys()):
                    win_utils.draw_rect(x, y, self.PIECE_COLOUR_DICT[id], surface)
                    
    def check_game_over(self):
        for y in range(bc.BOARD_HEIGHT_BUFFER):
            if any(id in self.PIECE_COLOUR_DICT.keys() for id in self.board_state[y].tolist()):
                return True
        return False
    
    def get_max_height_column_list(self):
        max_height_list = []
        
        # Horizontal loop
        for i in range(bc.BOARD_COLUMNS):
            column = self.board_state[:, i]
            
            peak_found = False
            
            # Vertical loop
            for j in bc.BOARD_HEIGHT_RANGE_INCR:
                if (column[j] > 0):
                    max_height_list.append(bc.BOARD_HEIGHT - j)
                    peak_found = True
                    break
                
            if (peak_found == False):
                max_height_list.append(0) 
        
        return max_height_list
    
    def get_max_height(self):
        return max(self.get_max_height_column_list())
    
    def get_min_height_column_list(self):
        min_height_list = []
        
        # Horizontal loop
        for i in range(bc.BOARD_COLUMNS):
            column = self.board_state[:, i]
            
            min_found = False
            
            # Vertical loop
            for j in bc.BOARD_HEIGHT_RANGE_INCR:
                if (column[j] > 0):
                    min_height_list.append(bc.BOARD_HEIGHT - j)
                    min_found = True
                    break
                
            if (min_found == False):
                min_height_list.append(0) 
        
        return min_height_list
    
    def get_min_height(self):
        return min(self.get_min_height_column_list())
    
    def get_num_of_gaps(self):
        max_height_list = self.get_max_height_column_list()
        gaps = 0
        
        # Horizontal loop
        for i in range(bc.BOARD_COLUMNS):
            column = self.board_state[:, i]

            # Vertical loop
            for j in range(bc.BOARD_HEIGHT - max_height_list[i], bc.BOARD_HEIGHT):
                if (column[j] == 0) and (column[j - 1] > 0):
                    gaps += 1
                    
        return gaps            
        
    def reset_board_state(self):
        self.board_state = self.INITIAL_BOARD_STATE.copy()
        self.occupied_spaces = 0
    
    def _init_board_state(self):
        return np.full(shape=(bc.BOARD_HEIGHT, bc.BOARD_WIDTH), fill_value=self.EMPTY_PIECE_ID)
    
    def get_minimal_board_state(self):
        min_board_state = self.board_state.copy()
        
        min_board_state = np.delete(min_board_state, range(0, bc.BOARD_HEIGHT_BUFFER), axis=0)
        
        return min_board_state