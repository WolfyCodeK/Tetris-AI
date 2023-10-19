import numpy as np

import tetraminos.rotation_transformations as rt
import utils.board_utils as bu

class Tetramino:
    GHOST_PIECE_ALPHA = 225
    
    def __init__(self, pid: chr, x: int, colour: tuple, kick_priority: dict, shape: np.ndarray) -> None:
        self.START_X_POS = x + bu.BOARD_STATE_WIDTH_BUFFER
        self.START_Y_POS = bu.MAX_PIECE_LENGTH
        
        self.pid = pid
        self.x_pos = self.START_X_POS
        self.y_pos = self.START_Y_POS
        self.colour = colour
        self.kick_priority = kick_priority
        self.shape = shape
        
        self.kick_options = list(range(0, 4))
        self.previous_pos = (self.x_pos, self.y_pos)
        
        self.rotation_state = 0
        self.rotation_direction = 1
        
        self.DEFAULT_SHAPE = shape.copy()
        self.previous_shape = shape.copy()
        self.minos = shape.copy()
        
        self.update_minos()
        
        self.active = True
    
    def draw(self, surface):
        for i in range(len(self.minos)):
            bu.draw_rect(self.minos[i][0], self.minos[i][1], self.colour, surface)
            
    def draw_ghost(self, surface, max_height):
        for i in range(len(self.minos)):
            bu.draw_rect(self.minos[i][0], self.minos[i][1] + max_height, (50, 50, 50, self.GHOST_PIECE_ALPHA), surface)
    
    def update_minos(self):      
        for i in range(len(self.shape)):
            self.minos[i][0] = self.shape[i][0] + self.x_pos
            self.minos[i][1] = self.shape[i][1] + self.y_pos
    
    def set_x_pos(self, x: int) -> None:
        self.x_pos = x
        self.update_minos()
    
    def set_y_pos(self, y: int) -> None:
        self.y_pos = y
        self.update_minos()
        
    def transform(self, x: int, y: int) -> None:
        self.set_x_pos(self.x_pos + x)
        self.set_y_pos(self.y_pos + y)
        
    def reset_pos(self):
        self.set_x_pos(self.START_X_POS)
        self.set_y_pos(self.START_Y_POS)
        
    def reset_shape(self):
        self.shape = self.DEFAULT_SHAPE.copy()
        self.update_minos() 
        
    def save_previous_pos(self):
        self.previous_pos = (self.x_pos, self.y_pos)
        
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
        
    def rotate(self, clockwise: bool, is_IPiece, is_OPiece):
        self.previous_shape = self.shape.copy()
        
        if (clockwise):
            self.shape = rt.rotate_clockwise(self.shape, is_IPiece, is_OPiece)
            self.rotation_direction = 1

        else:
            self.shape = rt.rotate_anticlockwise(self.shape, is_IPiece, is_OPiece)
            self.rotation_direction = -1
        
        if (not is_IPiece):
            self.update_rotation_state()
            
        self.update_minos()
    
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
    
    def revert_rotation(self, is_IPiece):
        self.shape = self.previous_shape.copy()
        self.rotation_direction = -self.rotation_direction
        
        if (not is_IPiece):
            self.update_rotation_state()
        
    def revert_kick(self):
        self.set_x_pos(self.previous_pos[0])
        self.set_y_pos(self.previous_pos[1])