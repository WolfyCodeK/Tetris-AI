from numpy import full
import board.board_definitions as bd
import board.board_utils as bu

from pieces.i_piece import IPiece
from pieces.o_piece import OPiece
from pieces.three_wide_pieces import (JPiece, LPiece, SPiece, TPiece, ZPiece)

class Board():
    EMPTY_PIECE_PID = 'E'
    
    # All the available pieces to the piece controller
    PIECE_LIST = [ZPiece, SPiece, JPiece, LPiece, TPiece, IPiece, OPiece]
    
    PIECE_PID_LIST  = []
    for i in range(len(PIECE_LIST)):
                PIECE_PID_LIST.append(PIECE_LIST[i].PID)
    
    COLOUR_PID_DICT = {}
    for i in range(len(PIECE_LIST)):
        COLOUR_PID_DICT[PIECE_PID_LIST[i]] = PIECE_LIST[i].COLOUR
    
    def __init__(self) -> None:
        self.INITIAL_BOARD_STATE = self._init_board_state()
        self.reset_board_state()
    
    def draw(self, surface):
        for y in range(bd.BOARD_HEIGHT_BUFFER, bd.BOARD_HEIGHT):
            for x in range(bd.BOARD_COLUMNS):
                pid = self.board_state[y][x]
                
                if (pid in Board.PIECE_PID_LIST):
                    bu.draw_rect(x, y, self.COLOUR_PID_DICT[pid], surface)
                    
    def check_game_over(self):
        for y in range(bd.BOARD_HEIGHT_BUFFER):
            if any(pid in self.PIECE_PID_LIST for pid in self.board_state[y].tolist()):
                return True
    
    def reset_board_state(self):
        self.board_state = self.INITIAL_BOARD_STATE.copy()
    
    def _init_board_state(self):
        # Initialise board state to be empty
        board_state = full(shape=(bd.BOARD_HEIGHT, bd.BOARD_COLUMNS), fill_value=self.EMPTY_PIECE_PID)
            
        return board_state