import board_utils as bd
import tetramino

class ZPiece(tetramino.Tetramino):
    
    colour = (255, 0, 0)
    
    def __init__(self, canvas) -> None:
        super().__init__(4, 0, canvas, self.colour)
    
    """
    def drop(self):
        if (self.y <= bd.BOARD_ROWS + bd.toGridSize(bd.BOARD_TOP_BUFFER) - bd.toGridSize(bd.DROP_HEIGHT)):
            self.y += 1
        else:
            self.active = False
    """
        
    
    def draw(self):
        x = self.x
        y = self.y

        super().draw_rect(x, y)
        self.draw_rect(x + 1, y)
        self.draw_rect(x + 1, y + 1)
        self.draw_rect(x + 2, y + 1)