from pygame import Surface
import board_utils as bu

class Tetramino:
    def __init__(self, x: int, y: int, colour: tuple, board_surface: Surface) -> None:
        self.x_pos = x
        self.y_pos = y
        self.colour = colour
        self.board_surface = board_surface
        
        self.active = True
    
    def draw(self):
        for i in range(len(self.occupying_squares)):
            bu.draw_rect(self.occupying_squares[i][0], self.occupying_squares[i][1], self.colour, self.board_surface)