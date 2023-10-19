from numpy import array
import pieces.rotation_transformations as rt

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
    
    def rotate(self, clockwise: bool):
        self.previous_shape = self.shape.copy()
        
        if (clockwise):
            self.shape = rt.rotate_clockwise(self.shape, is_IPiece=True)
            self.rotation_direction = 1

        else:
            self.shape = rt.rotate_anticlockwise(self.shape, is_IPiece=True)
            self.rotation_direction = -1
            
        self.update_minos()
    
    def get_kick_priority(self):
        if self.rotation_direction == 1:
            return self.CLOCKWISE_KICK_PRIORITY
        elif self.rotation_direction == -1:
            return self.ANTI_CLOCKWISE_KICK_PRIORITY 
        
    def kick(self, kick_index, clockwise):
        if clockwise:
            if (self.rotation_state == 0):
                if (kick_index == self.kick_options[0]):
                    self.transform(-2, 0)
                elif (kick_index == self.kick_options[1]):
                    self.transform(1, 0)
                elif (kick_index == self.kick_options[2]):
                    self.transform(-2, 1)
                elif (kick_index == self.kick_options[3]):
                    self.transform(1, -2)
                    
            elif (self.rotation_state == 1):
                if (kick_index == self.kick_options[0]):
                    self.transform(-1, 0)
                elif (kick_index == self.kick_options[1]):
                    self.transform(2, 0)
                elif (kick_index == self.kick_options[2]):
                    self.transform(-1, -2)
                elif (kick_index == self.kick_options[3]):
                    self.transform(2, 1)
                    
            elif (self.rotation_state == 2):
                if (kick_index == self.kick_options[0]):
                    self.transform(2, 0)
                elif (kick_index == self.kick_options[1]):
                    self.transform(-1, 0)
                elif (kick_index == self.kick_options[2]):
                    self.transform(2, -1)
                elif (kick_index == self.kick_options[3]):
                    self.transform(-1, 2)
                    
            elif (self.rotation_state == 3):
                if (kick_index == self.kick_options[0]):
                    self.transform(1, 0)
                elif (kick_index == self.kick_options[1]):
                    self.transform(-2, 0)
                elif (kick_index == self.kick_options[2]):
                    self.transform(1, 2)
                elif (kick_index == self.kick_options[3]):
                    self.transform(-2, -1)
        else:
            if (self.rotation_state == 0):
                if (kick_index == self.kick_options[0]):
                    self.transform(-1, 0)
                elif (kick_index == self.kick_options[1]):
                    self.transform(2, 0)
                elif (kick_index == self.kick_options[2]):
                    self.transform(-1, -2)
                elif (kick_index == self.kick_options[3]):
                    self.transform(2, 1)
                    
            elif (self.rotation_state == 1):
                if (kick_index == self.kick_options[0]):
                    self.transform(-2, 0)
                elif (kick_index == self.kick_options[1]):
                    self.transform(1, 0)
                elif (kick_index == self.kick_options[2]):
                    self.transform(-2, 1)
                elif (kick_index == self.kick_options[3]):
                    self.transform(1, -2)
                    
            elif (self.rotation_state == 2):
                if (kick_index == self.kick_options[0]):
                    self.transform(1, 0)
                elif (kick_index == self.kick_options[1]):
                    self.transform(-2, 0)
                elif (kick_index == self.kick_options[2]):
                    self.transform(1, 2)
                elif (kick_index == self.kick_options[3]):
                    self.transform(-2, -1)
                    
            elif (self.rotation_state == 3):
                if (kick_index == self.kick_options[0]):
                    self.transform(2, 0)
                elif (kick_index == self.kick_options[1]):
                    self.transform(-1, 0)
                elif (kick_index == self.kick_options[2]):
                    self.transform(2, -1)
                elif (kick_index == self.kick_options[3]):
                    self.transform(-1, 2)