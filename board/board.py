from numpy import full
import board.board_utils as bu

from pieces.i_piece import IPiece
from pieces.o_piece import OPiece
from pieces.symmetrical_piece import (JPiece, LPiece, SPiece, TPiece, ZPiece)

class Board():
    EMPTY_PIECE_PID = 'E'
    FLOOR_PIECE_PID = 'F'
    WALL_PIECE_PID = 'W'
    
    NONE_PIECE_TYPES = [EMPTY_PIECE_PID, FLOOR_PIECE_PID, WALL_PIECE_PID]
    
    # All the available pieces to the piece controller
    PIECE_LIST = [ZPiece, SPiece, JPiece, LPiece, TPiece, IPiece, OPiece]
    
    PIECE_PID_LIST  = []
    for i in range(len(PIECE_LIST)):
                PIECE_PID_LIST.append(PIECE_LIST[i].PID)
    
    BLOCKING_PIECE_TYPES = PIECE_PID_LIST.copy()
    BLOCKING_PIECE_TYPES.append(FLOOR_PIECE_PID)
    BLOCKING_PIECE_TYPES.append(WALL_PIECE_PID)
    
    COLOUR_PID_DICT = {}
    for i in range(len(PIECE_LIST)):
        COLOUR_PID_DICT[PIECE_PID_LIST[i]] = PIECE_LIST[i].COLOUR
    
    def __init__(self) -> None:
        self.INITIAL_BOARD_STATE = self._init_board_state()
        self.reset_board_state()
    
    def draw(self, surface):
        for y in range(len(self.board_state)):
            for x in range(len(self.board_state[0])):
                if (self.board_state[y][x] not in self.NONE_PIECE_TYPES):
                    bu.draw_rect(x, y, self.COLOUR_PID_DICT[self.board_state[y][x]], surface)
                    
    def check_game_over(self):
        for y in range(bu.BOARD_STATE_HEIGHT_BUFFER):
            if any(pid in self.PIECE_PID_LIST for pid in self.board_state[y].tolist()):
                return True
    
    def reset_board_state(self):
        self.board_state = self.INITIAL_BOARD_STATE.copy()
    
    def _init_board_state(self):
        # Initialise board state to be empty
        board_state = full(shape=(bu.BOARD_STATE_HEIGHT + bu.FLOOR_SIZE, bu.BOARD_STATE_WIDTH), fill_value=self.EMPTY_PIECE_PID)
            
        # Set wall pieces
        # Right wall
        for x in range(bu.BOARD_RIGHT_WALL, bu.BOARD_STATE_WIDTH):
            for y in range(len(board_state)):
                board_state[y][x] = self.WALL_PIECE_PID
            
        # Left wall
        for x in range(bu.BOARD_LEFT_WALL):
            for y in range(len(board_state)):
                board_state[y][x] = self.WALL_PIECE_PID
                
        # Set floor pieces
        for i in range(bu.FLOOR_SIZE):
            board_state[bu.BOARD_STATE_HEIGHT + i] = self.FLOOR_PIECE_PID
            
        return board_state