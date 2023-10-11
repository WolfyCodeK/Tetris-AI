import numpy as np
import board_utils as bu
from tetramino import Tetramino

class ZPiece(Tetramino):
    
    colour = (255, 0, 0)
    
    __x_pos = 4
    __y_pos = int(-(bu.pixel_to_grid_size(bu.DROP_HEIGHT)))
    
    def __init__(self, board_surface) -> None:
        super().__init__(self.__x_pos, self.__y_pos)
        self.board_surface = board_surface
        self.occupying_squares = self.__update_occupying_squares()
    
    def __update_occupying_squares(self):    
        return np.array([
            [self.__x_pos, self.__y_pos], 
            [self.__x_pos + 1, self.__y_pos],
            [self.__x_pos, self.__y_pos - 1],
            [self.__x_pos - 1, self.__y_pos - 1]
        ])
        
    def set_x_pos(self, x: int) -> None:
        self.__x_pos = x
        self.occupying_squares = self.__update_occupying_squares()
        
    def get_x_pos(self) -> int:
        return self.__x_pos
        
    def set_y_pos(self, y: int) -> None:
        self.__y_pos = y
        self.occupying_squares = self.__update_occupying_squares()
        
    def get_y_pos(self) -> int:
        return self.__y_pos
        
    def draw(self):
        for i in range(len(self.occupying_squares)):
            bu.draw_rect(self.occupying_squares[i][0], self.occupying_squares[i][1], self.colour, self.board_surface)