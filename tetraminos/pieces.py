import numpy as np

from .tetramino import Tetramino


class ZPiece(Tetramino):
    PID = 'Z'
    START_BOARD_X = 4
    COLOUR = (255,85,82)
    KICK_PRIORITY = {
        0: [0, 1, 2, 3],
        1: [2, 0, 1, 3],
        2: [1, 0, 2, 3],
        3: [3, 0, 1, 2]
    }
    
    def __init__(self) -> None:
        super().__init__(
            self.PID,
            self.START_BOARD_X, 
            self.COLOUR,
            self.KICK_PRIORITY,
            np.array([[0, 0], [1, 0], [0, -1], [-1, -1]])
        )

class LPiece(Tetramino):
    PID = 'L'
    START_BOARD_X = 4
    COLOUR = (255,159,122)
    KICK_PRIORITY = {
        0: [1, 0, 2, 3],
        1: [0, 3, 1, 2],
        2: [1, 0, 2, 3],
        3: [3, 0, 1, 2]
    }
    
    def __init__(self) -> None:
        super().__init__(
            self.PID,
            self.START_BOARD_X, 
            self.COLOUR,
            self.KICK_PRIORITY,
            np.array([[0, 0], [-1, 0], [1, 0], [1, -1]])
        )
        
class SPiece(Tetramino):
    PID = 'S'
    START_BOARD_X = 4
    COLOUR = (82,255,97)
    KICK_PRIORITY = {
        0: [0, 1, 2, 3],
        1: [3, 0, 1, 2],
        2: [1, 0, 2, 3],
        3: [2, 0, 1, 3]
    }
    
    def __init__(self) -> None:
        super().__init__(
            self.PID,
            self.START_BOARD_X, 
            self.COLOUR,
            self.KICK_PRIORITY,
            np.array([[0, 0], [-1, 0], [0, -1], [1, -1]])
        )
        
class JPiece(Tetramino):
    PID = 'J'
    START_BOARD_X = 4
    COLOUR = (62,101,255)
    KICK_PRIORITY = {
        0: [1, 0, 2, 3],
        1: [0, 2, 1, 3],
        2: [1, 0, 2, 3],
        3: [3, 0, 1, 2]
    }
    
    def __init__(self) -> None:
        super().__init__(
            self.PID,
            self.START_BOARD_X, 
            self.COLOUR,
            self.KICK_PRIORITY,
            np.array([[0, 0], [-1, 0], [1, 0], [-1, -1]])
        )
        
class TPiece(Tetramino):
    PID = 'T'
    START_BOARD_X = 4
    COLOUR = (255,100,167)
    KICK_PRIORITY = {
        0: [1, 0, 2, 3],
        1: [3, 0, 1, 2],
        2: [1, 0, 2, 3],
        3: [3, 2, 1, 0]
    }
    
    def __init__(self) -> None:
        super().__init__(
            self.PID,
            self.START_BOARD_X, 
            self.COLOUR,
            self.KICK_PRIORITY,
            np.array([[0, 0], [-1, 0], [1, 0], [0, -1]])
        )
    
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
        
class OPiece(Tetramino):
    PID = 'O'
    START_BOARD_X = 5
    COLOUR = (255,255,102)
    
    def __init__(self) -> None:
        super().__init__(
            self.PID,
            self.START_BOARD_X, 
            self.COLOUR,
            None,
            np.array([[0, 0], [-1, 0], [-1, -1], [0, -1]])
        )