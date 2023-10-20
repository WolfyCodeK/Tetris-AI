from numpy import array, ndarray
from pieces.piece_lookup_tables import IPIECE_ROTATION_TABLE, IPIECE_KICK_TABLE

from .piece import Piece

class IPiece(Piece):
    PID = 'I'
    START_BOARD_X = 4
    COLOUR = (122,161,255)
    DEFAULT_SHAPE = array([[0, 0], [-1, 0], [1, 0], [2, 0]])
    CLOCKWISE_KICK_PRIORITY = {
        0: [2, 0, 1, 3],
        1: [2, 0, 1, 3],
        2: [3, 0, 1, 2],
        3: [0, 1, 2, 3]
    }
    
    ANTI_CLOCKWISE_KICK_PRIORITY = {
        0: [0, 1, 2, 3],
        1: [2, 0, 1, 3],
        2: [2, 0, 1, 3],
        3: [3, 0, 1, 2]
    }
    
    def __init__(self) -> None:
        super().__init__(self.PID, self.START_BOARD_X, self.COLOUR, self.DEFAULT_SHAPE.copy())
    
    def _adjust_i_piece(self, clockwise: bool, shape: ndarray, state: int, i: int):
        if state in [0, 2]:
            j = 0
        else:
            j = 1
        
        if not clockwise:
            state += 4
        
        piece_num = shape[i][j] + 1
        
        shape[i][0] = shape[i][0] + IPIECE_ROTATION_TABLE[state][piece_num][0]
        shape[i][1] = shape[i][1] + IPIECE_ROTATION_TABLE[state][piece_num][1]
        
        return shape
    
    def rotate(self, clockwise: bool):
        self.previous_shape = self.shape.copy()
        
        if clockwise:
            self.rotation_direction = 1
        else:
            self.rotation_direction = -1
        
        if (self.shape[0][0] == 0 and self.shape[0][1] == 0): # STATE 0
            for i in range(len(self.shape)):
                self.shape = self._adjust_i_piece(clockwise, self.shape, 0, i)
                    
        elif (self.shape[0][0] == 1 and self.shape[0][1] == 0): # STATE 1
            for i in range(len(self.shape)):
                self.shape = self._adjust_i_piece(clockwise, self.shape, 1, i)
                
        elif (self.shape[0][0] == 1 and self.shape[0][1] == 1): # STATE 2
            for i in range(len(self.shape)):
                self.shape = self._adjust_i_piece(clockwise, self.shape, 2, i)
                
        elif (self.shape[0][0] == 0 and self.shape[0][1] == 1): # STATE 3
            for i in range(len(self.shape)):
                self.shape = self._adjust_i_piece(clockwise, self.shape, 3, i)
            
        self.update_minos()
    
    def get_kick_priority(self):
        if self.rotation_direction == 1:
            return self.CLOCKWISE_KICK_PRIORITY
        elif self.rotation_direction == -1:
            return self.ANTI_CLOCKWISE_KICK_PRIORITY 
        
    def kick(self, kick_index, clockwise):
        relative_rot_state = self.rotation_state
        
        if not clockwise:
            relative_rot_state += 4
            
        self.transform(
            IPIECE_KICK_TABLE[relative_rot_state][kick_index][0], 
            IPIECE_KICK_TABLE[relative_rot_state][kick_index][1]
        )
        
        print(IPIECE_KICK_TABLE[relative_rot_state][kick_index][0])
        print(IPIECE_KICK_TABLE[relative_rot_state][kick_index][1])