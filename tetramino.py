import pygame
import board_utils as bd

class Tetramino:
    def __init__(self, x: int, y: int, canvas: pygame.Surface, colour: tuple) -> None:
        self.x = x
        self.y = y
        self.canvas = canvas
        self.colour = colour
        
        self.active = True
        
    def draw_rect(self, x, y):
        x = x + bd.BOARD_LEFT_BUFFER / bd.GRID_SIZE
        y = y + bd.BOARD_TOP_BUFFER / bd.GRID_SIZE - (bd.DROP_HEIGHT / bd.GRID_SIZE)
        pygame.draw.rect(self.canvas, self.colour, pygame.Rect((x - 1) * bd.GRID_SIZE, (y - 1) * bd.GRID_SIZE, bd.GRID_SIZE, bd.GRID_SIZE))