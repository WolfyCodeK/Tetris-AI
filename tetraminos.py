import pygame
import board_definitions as bd

class zTetromino:
    def __init__(self, x: int, y: int, canvas) -> None:
        self.x = x
        self.y = y
        self.canvas = canvas

    def __drawRect(self, x, y):
        x = x + bd.BOARD_LEFT_BUFFER / bd.GRID_SIZE
        y = y + bd.BOARD_TOP_BUFFER / bd.GRID_SIZE - (bd.DROP_HEIGHT / bd.GRID_SIZE)
        pygame.draw.rect(self.canvas, (255, 0, 0), pygame.Rect((x - 1) * bd.GRID_SIZE, (y - 1) * bd.GRID_SIZE, bd.GRID_SIZE, bd.GRID_SIZE))
    
    def drop(self):
        if (self.y <= bd.BOARD_ROWS + bd.toGridSize(bd.BOARD_TOP_BUFFER) - bd.toGridSize(bd.DROP_HEIGHT)):
            self.y += 1
    
    def draw(self):
        x = self.x
        y = self.y

        self.__drawRect(x, y)
        self.__drawRect(x + 1, y)
        self.__drawRect(x + 1, y + 1)
        self.__drawRect(x + 2, y + 1)