from numpy import full
import board.board_definitions as bd
import board.board_utils as bu

from pieces.i_piece import IPiece
from pieces.o_piece import OPiece
from pieces.three_wide_pieces import (JPiece, LPiece, SPiece, TPiece, ZPiece)

class Board():
    # All the available pieces to the piece controller
    PIECE_LIST = [ZPiece, SPiece, JPiece, LPiece, TPiece, IPiece, OPiece]
    
    EMPTY_PIECE_ID = 'E'
    
    # Dict of (ID, colour) pairs for all pieces
    PIECE_COLOUR_DICT = {}
    for i in range(len(PIECE_LIST)):
        PIECE_COLOUR_DICT[PIECE_LIST[i].ID] = PIECE_LIST[i].COLOUR
    
    def __init__(self) -> None:
        self.INITIAL_BOARD_STATE = self._init_board_state()
        self.reset_board_state()
    
    def draw(self, surface):
        for y in bd.BOARD_HEIGHT_RANGE:
            for x in range(bd.BOARD_WIDTH):
                id = self.board_state[y][x]
                
                if (id in Board.PIECE_COLOUR_DICT.keys()):
                    bu.draw_rect(x, y, self.PIECE_COLOUR_DICT[id], surface)
                    
    def check_game_over(self):
        for y in range(bd.BOARD_HEIGHT_BUFFER):
            if any(id in self.PIECE_COLOUR_DICT.keys() for id in self.board_state[y].tolist()):
                return True
    
    def reset_board_state(self):
        self.board_state = self.INITIAL_BOARD_STATE.copy()
    
    def _init_board_state(self):
        return full(shape=(bd.BOARD_HEIGHT, bd.BOARD_WIDTH), fill_value=self.EMPTY_PIECE_ID)