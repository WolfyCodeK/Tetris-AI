import board_utils as bu
import numpy as np

class Tetramino:
    def __init__(self, pid: chr, x: int, colour: tuple, large_rotation: bool = False) -> None:
        self.pid = pid
        self.x_pos = x
        self.y_pos = bu.PIECE_START_HEIGHT
        self.colour = colour
        
        self.rotational_space = np.array([
            [[0,0], [1,0], [2,0]],
            [[0,1], [1,1], [2,1]],
            [[0,2], [1,2], [2,2]]
        ])
        
        if (large_rotation):
            self.rotational_space = np.array([
                [[0,0], [1,0], [2,0], [3,0]],
                [[0,1], [1,1], [2,1], [3,1]],
                [[0,2], [1,2], [2,2], [3,2]],
                [[0,2], [1,2], [2,2], [3,3]]
            ])
        
        self.active = True
    
    def draw(self, board_surface):
        for i in range(len(self.occupying_squares)):
            bu.draw_rect(self.occupying_squares[i][0], self.occupying_squares[i][1], self.colour, board_surface)
    
    def update_occupying_squares(self, x, y):      
        raise NotImplementedError("Child class must override update_occupying_squares")
    
    def set_x_pos(self, x: int) -> None:
        self.x_pos = x
        self.update_occupying_squares(self.x_pos, self.y_pos)
    
    def set_y_pos(self, y: int) -> None:
        self.y_pos = y
        self.update_occupying_squares(self.x_pos, self.y_pos)
    
    def rotate_anticlockwise(self):
        pass
    
    def rotate_clockwise(self):
        print(self.rotational_space)