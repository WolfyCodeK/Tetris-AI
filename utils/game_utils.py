from pieces.piece_type_id import PieceTypeID
from utils.board_constants import *
import numpy as np

from controllers.game_controller import GameController

def get_bumpiness(game_controller: GameController):
    ...
    
def does_I_dependency_exist(game_controller: GameController):
    ...
    
def does_none_central_I_piece_exist(game_controller: GameController):
    ...

def get_max_height_column_list(game_controller: GameController):    
    board_state = game_controller.piece_manager.board.board_state
    
    max_height_list = []
    
    # Horizontal loop
    for i in range(BOARD_COLUMNS):
        column = board_state[:, i]
        
        peak_found = False
        
        # Vertical loop
        for j in BOARD_HEIGHT_RANGE_INCR:
            if (column[j] > 0):
                max_height_list.append(BOARD_HEIGHT - j)
                peak_found = True
                break
            
        if (peak_found == False):
            max_height_list.append(0) 
    
    return max_height_list

def get_max_height(game_controller: GameController):
    return max(get_max_height_column_list(game_controller))

def get_min_height_column_list(game_controller: GameController):
    board_state = game_controller.piece_manager.board.board_state
    
    min_height_list = []
    
    # Horizontal loop
    for i in range(BOARD_COLUMNS):
        column = board_state[:, i]
        
        min_found = False
        
        # Vertical loop
        for j in BOARD_HEIGHT_RANGE_INCR:
            if (column[j] > 0):
                min_height_list.append(BOARD_HEIGHT - j)
                min_found = True
                break
            
        if (min_found == False):
            min_height_list.append(0) 
    
    return min_height_list

def get_min_piece_height_on_board(game_controller: GameController):
    return min(get_min_height_column_list(game_controller))

def get_first_gap_list(game_controller: GameController):
    board_state = game_controller.piece_manager.board.board_state
    
    first_gap_list = []
    
    # Horizontal loop
    for i in range(BOARD_COLUMNS):
        column = board_state[:, i]
        
        # Vertical loop
        for j in range(BOARD_HEIGHT - 1, BOARD_HEIGHT_BUFFER, -1):
            if (column[j] == 0):
                first_gap_list.append((BOARD_HEIGHT - 1) - j)
                break
    
    return first_gap_list

def get_num_of_top_gaps(game_controller: GameController):
    board_state = game_controller.piece_manager.board.board_state
    
    max_height_list = get_max_height_column_list(game_controller)
    gaps = 0
    
    # Horizontal loop
    for i in range(BOARD_COLUMNS):
        column = board_state[:, i]
        left_wall = False
        right_wall = False
        
        if i > 0:
            left = board_state[:, i - 1]
        else:
            left_wall = True
            
        if i < BOARD_COLUMNS - 1:   
            right = board_state[:, i + 1]
        else:
            right_wall = True

        # Vertical loop
        for j in range(BOARD_HEIGHT - max_height_list[i], BOARD_HEIGHT):
            if (column[j] == 0) and (column[j - 1] > 0) and ((not((left_wall == True) or (left[j] > 0))) or (not((right_wall == True) or (right[j] > 0)))):
                gaps += 1
                
    return gaps      

def get_num_of_full_gaps(game_controller: GameController):
    board_state = game_controller.piece_manager.board.board_state
    
    max_height = get_max_height(game_controller)
    gaps = 0
    
    # Vertical loop
    for i in range(BOARD_HEIGHT - max_height, BOARD_HEIGHT):
        middle_row = board_state[i, :]
        
        if (i > 0):
            top_row = board_state[i - 1, :]
        else:
            top_row = None
        
        # Horizontal loop
        for j in range(BOARD_COLUMNS):
            if (middle_row[j] == 0) and (top_row[j] > 0) and ((j == 0) or (middle_row[j - 1] > 0)):
                if (j == BOARD_COLUMNS - 1) or (middle_row[j + 1] > 0):
                    added_gaps = 1
                else:
                    check_finished = False
                    
                    added_gaps = 1
                    
                    while not check_finished:
                        j += 1
                        
                        if (j < BOARD_COLUMNS):
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

def check_row_filled(game_controller: GameController, row):
    board_state = game_controller.piece_manager.board.board_state
    
    filled_count = 0
    
    for i in range(BOARD_COLUMNS):
        if board_state[row, i] > 0:
            filled_count += 1
            
    if filled_count == BOARD_ROWS - 1:
        return True
    else:
        return False
    
def check_all_previous_rows_filled(game_controller: GameController):
    all_rows_filled = False
    top_filled_found = False
    previous_filled = True
    row = BOARD_HEIGHT - 1
    
    while not top_filled_found:
        if check_row_filled(game_controller, row) and not previous_filled:
            all_rows_filled = False
            top_filled_found = True
            
        elif check_row_filled(game_controller, row) and previous_filled:
            row -= 1
            all_rows_filled = True
            previous_filled = True
            
        elif not check_row_filled(game_controller, row) and previous_filled:
            row -= 1
            previous_filled = False
            all_rows_filled = True
            top_filled_found = False
            
        else:
            top_filled_found = True
            all_rows_filled = True
            
    return all_rows_filled

def get_minimal_board_state(game_controller: GameController):
    board_state = game_controller.piece_manager.board.board_state
    
    min_board_state = board_state.copy()
    
    min_board_state = np.delete(min_board_state, range(0, BOARD_HEIGHT_BUFFER), axis=0)
    
    return min_board_state

def is_tetris_ready(game_controller: GameController):
    max_list = list(get_max_height_column_list(game_controller).copy())
    max_list.pop()
    
    return not any([height < 4 for height in max_list])

def is_well_valid(game_controller: GameController):
    board_state = game_controller.piece_manager.board.board_state
    
    valid = True
    
    # Vertical loop
    for i in BOARD_HEIGHT_RANGE_INCR:
        last_column_id = board_state[i, BOARD_COLUMNS - 1]
        
        if last_column_id != PieceTypeID.I_PIECE and last_column_id != PieceTypeID.EMPTY:
            valid = False
        elif last_column_id == PieceTypeID.I_PIECE and not is_tetris_ready(game_controller): 
            valid = False
        
    return valid
    
def get_second_lowest_gap(game_controller: GameController):
    return sorted(get_first_gap_list(game_controller))[1]    
    
def get_max_piece_height_on_board(game_controller: GameController):
    return get_max_height(game_controller)

def get_board_height_difference_with_well(game_controller: GameController):
    return get_max_piece_height_on_board(game_controller) - get_min_gap_height_exluding_well(game_controller)

def get_truncated_piece_queue(game_controller: GameController, first_n_pieces):
    return game_controller.piece_manager.piece_queue.get_truncated_piece_queue(first_n_pieces)

def get_current_piece_id(game_controller: GameController) -> int:
    return game_controller.piece_manager.current_piece.id

def get_piece_value_bounds(game_controller: GameController):
    return game_controller.piece_manager.board.EMPTY_PIECE_ID, len(PieceTypeID)

def get_held_piece_id(game_controller: GameController) -> int:
    return game_controller.piece_manager.piece_holder.held_piece.id if game_controller.piece_manager.piece_holder.held_piece is not None else 0

def get_min_gap_height_exluding_well(game_controller: GameController) -> int:
    gap_list = get_first_gap_list(game_controller).copy()
    
    # Remove well value
    gap_list.pop()
    
    return sorted(gap_list)[0]