from numpy import ndarray, array
import tetraminos.rotation_transformations as rt

from .tetramino import Tetramino


class SymmetricalTetramino(Tetramino):
    def __init__(self, pid: chr, x: int, colour: tuple, kick_priority: dict, shape: ndarray) -> None:
        super().__init__(pid, x, colour, shape)
        self.kick_priority = kick_priority
    
    def rotate(self, clockwise: bool):
        self.previous_shape = self.shape.copy()
        
        if (clockwise):
            self.shape = rt.rotate_clockwise(self.shape)
            self.rotation_direction = 1

        else:
            self.shape = rt.rotate_anticlockwise(self.shape)
            self.rotation_direction = -1
            
        self.update_rotation_state()  
        self.update_minos()
        
    def update_rotation_state(self):
        if self.rotation_direction == 1:
            if (self.rotation_state == 3):
                self.rotation_state = 0
            else:
                self.rotation_state += 1
        elif self.rotation_direction == -1:
            if (self.rotation_state == 0):
                self.rotation_state = 3
            else:
                self.rotation_state -= 1
    
    def get_kick_priority(self):
        return self.kick_priority
        
    def kick(self, kick_index, clockwise):
        relative_rot_state = self.rotation_state
        
        if clockwise:
            rot = 1
        else:
            # Invert horizonal transformations
            rot = -1
            
            # Mirror clockwise transformations
            if (self.rotation_state in [0, 1, 2]):
                relative_rot_state = relative_rot_state + 1
                
            if (self.rotation_state == 3):
                relative_rot_state = 0

        if (relative_rot_state == 0):
            if (kick_index == self.kick_options[0]):
                self.transform(rot * -1, rot * 0)
            elif (kick_index == self.kick_options[1]):
                self.transform(rot * -1, rot * 1)
            elif (kick_index == self.kick_options[2]):
                self.transform(rot * 0, rot * -2)
            elif (kick_index == self.kick_options[3]):
                self.transform(rot * -1, rot * -2)
                
        elif (relative_rot_state == 1):
            if (kick_index == self.kick_options[0]):
                self.transform(rot * -1, rot * 0)
            elif (kick_index == self.kick_options[1]):
                self.transform(rot * -1, rot * -1)
            elif (kick_index == self.kick_options[2]):
                self.transform(rot * 0, rot * 2)
            elif (kick_index == self.kick_options[3]):
                self.transform(rot * -1, rot * 2)
                
        elif (relative_rot_state == 2):
            if (kick_index == self.kick_options[0]):
                self.transform(rot * 1, rot * 0)
            elif (kick_index == self.kick_options[1]):
                self.transform(rot * 1, rot * -1)
            elif (kick_index == self.kick_options[2]):
                self.transform(rot * 0, rot * -2)
            elif (kick_index == self.kick_options[3]):
                self.transform(rot * 1, -2)
                
        elif (relative_rot_state == 3):
            if (kick_index == self.kick_options[0]):
                self.transform(rot * 1, rot * 0)
            elif (kick_index == self.kick_options[1]):
                self.transform(rot * 1, rot * -1)
            elif (kick_index == self.kick_options[2]):
                self.transform(rot * 0, rot * 2)
            elif (kick_index == self.kick_options[3]):
                self.transform(rot * 1, rot * 2)
        
    def revert_rotation(self):
        self.shape = self.previous_shape.copy()
        self.rotation_direction = -self.rotation_direction
        self.update_rotation_state()
        
class ZPiece(SymmetricalTetramino):
    PID = 'Z'
    START_BOARD_X = 4
    COLOUR = (255,85,82)
    SHAPE = array([[0, 0], [1, 0], [0, -1], [-1, -1]])
    KICK_PRIORITY = {
        0: [0, 1, 2, 3],
        1: [2, 0, 1, 3],
        2: [1, 0, 2, 3],
        3: [3, 0, 1, 2]
    }
    
    def __init__(self) -> None:
        super().__init__(self.PID, self.START_BOARD_X, self.COLOUR, self.KICK_PRIORITY, self.SHAPE)

class LPiece(SymmetricalTetramino):
    PID = 'L'
    START_BOARD_X = 4
    COLOUR = (255,159,122)
    SHAPE = array([[0, 0], [-1, 0], [1, 0], [1, -1]])
    KICK_PRIORITY = {
        0: [1, 0, 2, 3],
        1: [0, 3, 1, 2],
        2: [1, 0, 2, 3],
        3: [3, 0, 1, 2]
    }
    
    def __init__(self) -> None:
        super().__init__(self.PID, self.START_BOARD_X, self.COLOUR, self.KICK_PRIORITY, self.SHAPE)
        
class SPiece(SymmetricalTetramino):
    PID = 'S'
    START_BOARD_X = 4
    COLOUR = (82,255,97)
    SHAPE = array([[0, 0], [-1, 0], [0, -1], [1, -1]])
    KICK_PRIORITY = {
        0: [0, 1, 2, 3],
        1: [3, 0, 1, 2],
        2: [1, 0, 2, 3],
        3: [2, 0, 1, 3]
    }
    
    def __init__(self) -> None:
        super().__init__(self.PID, self.START_BOARD_X, self.COLOUR, self.KICK_PRIORITY, self.SHAPE)
        
class JPiece(SymmetricalTetramino):
    PID = 'J'
    START_BOARD_X = 4
    COLOUR = (62,101,255)
    SHAPE = array([[0, 0], [-1, 0], [1, 0], [-1, -1]])
    KICK_PRIORITY = {
        0: [1, 0, 2, 3],
        1: [0, 2, 1, 3],
        2: [1, 0, 2, 3],
        3: [3, 0, 1, 2]
    }
    
    def __init__(self) -> None:
        super().__init__(self.PID, self.START_BOARD_X, self.COLOUR, self.KICK_PRIORITY, self.SHAPE)
        
class TPiece(SymmetricalTetramino):
    PID = 'T'
    START_BOARD_X = 4
    COLOUR = (255,100,167)
    SHAPE = array([[0, 0], [-1, 0], [1, 0], [0, -1]])
    KICK_PRIORITY = {
        0: [1, 0, 2, 3],
        1: [3, 0, 1, 2],
        2: [1, 0, 2, 3],
        3: [3, 2, 1, 0]
    }
    
    def __init__(self) -> None:
        super().__init__(self.PID, self.START_BOARD_X, self.COLOUR, self.KICK_PRIORITY, self.SHAPE)
    
    def kick(self, kick_index, clockwise): 
        relative_rot_state = self.rotation_state
        
        if clockwise:
            rot = 1
        else:
            # Invert horizonal transformations
            rot = -1
            
            # Mirror clockwise transformations
            if (self.rotation_state in [0, 1, 2]):
                relative_rot_state = relative_rot_state + 1
                
            elif (self.rotation_state == 3):
                relative_rot_state = 0

        if (relative_rot_state == 0):
            if (kick_index == self.kick_options[0]):
                self.transform(rot * -1, rot * 0)
            elif (kick_index == self.kick_options[1]):
                self.transform(rot * -1, rot * 1)
            elif (kick_index == self.kick_options[2]):
                self.transform(rot * 0, rot * -2)
            elif (kick_index == self.kick_options[3]):
                self.transform(rot * -1, -2)
                
        elif (relative_rot_state == 1):
            if (kick_index == self.kick_options[0]):
                self.transform(rot * -1, rot * 0)
            elif (kick_index == self.kick_options[1]):
                self.transform(rot * -1, rot * -1)
            elif (kick_index == self.kick_options[3]):
                self.transform(rot * -1, rot * 2)
                
        elif (relative_rot_state == 2):
            if (kick_index == self.kick_options[0]):
                self.transform(rot * 1, rot * 0)
            elif (kick_index == self.kick_options[1]):
                self.transform(rot * 1, rot * -1)
            elif (kick_index == self.kick_options[2]):
                self.transform(rot * 0, rot * -2)
            elif (kick_index == self.kick_options[3]):
                self.transform(rot * 1, rot * -2)
                
        elif (relative_rot_state == 3):
            if (kick_index == self.kick_options[0]):
                self.transform(rot * 1, rot * 0)
            elif (kick_index == self.kick_options[2]):
                self.transform(rot * 0, rot * 2)
            elif (kick_index == self.kick_options[3]):
                self.transform(rot * 1, rot * 2)