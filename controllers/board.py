import numpy as np
from pieces.piece_type_id import PieceTypeID
import utils.board_constants as bc

import utils.window_utils as win_utils
from pieces.i_piece import IPiece
from pieces.o_piece import OPiece
from pieces.three_wide_pieces import (JPiece, LPiece, SPiece, TPiece, ZPiece)

class Board():
    # All the available pieces to the piece controller
    PIECE_LIST = [IPiece, OPiece, ZPiece, SPiece, JPiece, LPiece, TPiece]
    
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
    
    def get_first_gap_list(self):
        first_gap_list = []
        
        # Horizontal loop
        for i in range(bc.BOARD_COLUMNS):
            column = self.board_state[:, i]
            
            # Vertical loop
            for j in range(bc.BOARD_HEIGHT - 1, bc.BOARD_HEIGHT_BUFFER, -1):
                if (column[j] == 0):
                    first_gap_list.append((bc.BOARD_HEIGHT - 1) - j)
                    break
        
        return first_gap_list
    
    def get_num_of_top_gaps(self):
        max_height_list = self.get_max_height_column_list()
        gaps = 0
        
        # Horizontal loop
        for i in range(bc.BOARD_COLUMNS):
            column = self.board_state[:, i]
            left_wall = False
            right_wall = False
            
            if i > 0:
                left = self.board_state[:, i - 1]
            else:
                left_wall = True
                
            if i < bc.BOARD_COLUMNS - 1:   
                right = self.board_state[:, i + 1]
            else:
                right_wall = True

            # Vertical loop
            for j in range(bc.BOARD_HEIGHT - max_height_list[i], bc.BOARD_HEIGHT):
                if (column[j] == 0) and (column[j - 1] > 0) and ((not((left_wall == True) or (left[j] > 0))) or (not((right_wall == True) or (right[j] > 0)))):
                    gaps += 1
                    
        return gaps      
    
    def get_num_of_full_gaps(self):
        max_height = self.get_max_height()
        gaps = 0
        
        # Vertical loop
        for i in range(bc.BOARD_HEIGHT - max_height, bc.BOARD_HEIGHT):
            middle_row = self.board_state[i, :]
            
            if (i > 0):
                top_row = self.board_state[i - 1, :]
            else:
                top_row = None
            
            # Horizontal loop
            for j in range(bc.BOARD_COLUMNS):
                if (middle_row[j] == 0) and (top_row[j] > 0) and ((j == 0) or (middle_row[j - 1] > 0)):
                    if (j == bc.BOARD_COLUMNS - 1) or (middle_row[j + 1] > 0):
                        added_gaps = 1
                    else:
                        check_finished = False
                        
                        added_gaps = 1
                        
                        while not check_finished:
                            j += 1
                            
                            if (j < bc.BOARD_COLUMNS):
                                next_pos_right = middle_row[j] == 0
                                next_pos_top = top_row[j] > 0
                                
                                if next_pos_right and not next_pos_top:
                                    check_finished = True
                                    added_gaps = 0
                                    
                                elif next_pos_right and next_pos_top:
                                    added_gaps += 1
                                    
                                elif not next_pos_right:
                                    check_finished = True
                            else:
                                check_finished = True
                                
                    gaps += added_gaps

        return gaps      
    
    def check_row_filled(self, row):
        filled_count = 0
        
        for i in range(bc.BOARD_COLUMNS):
            if self.board_state[row, i] > 0:
                filled_count += 1
                
        if filled_count == bc.BOARD_ROWS - 1:
            return True
        else:
            return False
        
    def check_all_previous_rows_filled(self):
        all_rows_filled = False
        top_filled_found = False
        previous_filled = True
        row = bc.BOARD_HEIGHT - 1
        
        while not top_filled_found:
            if self.check_row_filled(row) and not previous_filled:
                all_rows_filled = False
                top_filled_found = True
                
            elif self.check_row_filled(row) and previous_filled:
                row -= 1
                all_rows_filled = True
                previous_filled = True
                
            elif not self.check_row_filled(row) and previous_filled:
                row -= 1
                previous_filled = False
                all_rows_filled = True
                top_filled_found = False
                
            else:
                top_filled_found = True
                all_rows_filled = True
                
        return all_rows_filled
        
    def reset_board_state(self):
        self.board_state = self.INITIAL_BOARD_STATE.copy()
        self.occupied_spaces = 0
    
    def _init_board_state(self):
        return np.full(shape=(bc.BOARD_HEIGHT, bc.BOARD_WIDTH), fill_value=self.EMPTY_PIECE_ID)
    
    def get_minimal_board_state(self):
        min_board_state = self.board_state.copy()
        
        min_board_state = np.delete(min_board_state, range(0, bc.BOARD_HEIGHT_BUFFER), axis=0)
        
        return min_board_state
    
    def is_tetris_ready(self):
        max_list = self.get_max_height_column_list().copy()
        max_list.pop()
        
        return not any([height < 4 for height in max_list])
    
    def is_well_valid(self):
        valid = True
        
        max_list = self.get_max_height_column_list().copy()
        max_list.pop()
        
        # Vertical loop
        for i in bc.BOARD_HEIGHT_RANGE_INCR:
            last_column_id = self.board_state[i, bc.BOARD_COLUMNS - 1]
            
            if last_column_id != PieceTypeID.I_PIECE and last_column_id != PieceTypeID.EMPTY:
                valid = False
            elif last_column_id == PieceTypeID.I_PIECE and not self.is_tetris_ready(): 
                valid = False
            
        return valid