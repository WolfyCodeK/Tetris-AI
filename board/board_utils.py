from pygame import Rect, Surface, draw
import board.board_definitions as bd
from board.board import Board

def pixel_to_grid_size(x: int) -> int:
    """Converts pixel size to grid size e.g. 160 pixels to 5 squares, where grid size = 32.

    Args:
        x (int): Any length of pixels

    Returns:
        int: That length divided by the grid size
    """    
    return x / Board.GRID_SIZE

def grid_to_pixel_length(x: int) -> int:
    """Converts grid size to pixel size e.g. 5 squares to 160 pixels, where grid size = 32.

    Args:
        x (int): Any number of squares

    Returns:
        int: That length multiplied by the grid size
    """
    return x * Board.GRID_SIZE

def _draw_outer_grid(surface: Surface) -> None:
    """Draw the thick outline of the boards grid.

    Args:
        surface (Surface): The surface being drawn to.
    """
    # BOTTOM
    draw.line(surface, (255, 255, 255, bd.OUTER_GRID_ALPHA), bd.BOTTOM_LEFT_BOARD_CORNER, bd.BOTTOM_RIGHT_BOARD_CORNER, width=bd.GRID_OUTLINE_WIDTH)
    
    # LEFT
    draw.line(surface, (255, 255, 255, bd.OUTER_GRID_ALPHA), bd.TOP_LEFT_BOARD_CORNER, bd.BOTTOM_LEFT_BOARD_CORNER, width=bd.GRID_OUTLINE_WIDTH)
    
    # RIGHT
    draw.line(surface, (255, 255, 255, bd.OUTER_GRID_ALPHA), bd.TOP_RIGHT_BOARD_CORNER, bd.BOTTOM_RIGHT_BOARD_CORNER, width=bd.GRID_OUTLINE_WIDTH)
    
def _draw_inner_grid(surface: Surface) -> None:
    """Draw the thin inner lines of the boards grid.

    Args:
        surface (Surface): The surface being drawn to.
    """
    # VERTICAL LINES
    for x in range(1 * Board.GRID_SIZE, bd.BOARD_PIXEL_WIDTH, Board.GRID_SIZE):
        draw.line(
            surface, 
            (255, 255, 255, bd.INNER_GRID_ALPHA), 
            (x + bd.BOARD_LEFT_BUFFER, bd.BOARD_TOP_BUFFER + 2), 
            (x + bd.BOARD_LEFT_BUFFER, bd.BOARD_PIXEL_HEIGHT + bd.BOARD_TOP_BUFFER - 1)
        )
    
    # HORIZONTAL LINES    
    for y in range(1 * Board.GRID_SIZE, bd.BOARD_PIXEL_HEIGHT, Board.GRID_SIZE):
        draw.line(
            surface, 
            (255, 255, 255, bd.INNER_GRID_ALPHA), 
            (bd.BOARD_LEFT_BUFFER, y + bd.BOARD_TOP_BUFFER), 
            (bd.BOARD_PIXEL_WIDTH + bd.BOARD_LEFT_BUFFER - 1, y + bd.BOARD_TOP_BUFFER)
        )

def draw_grids(surface: Surface, outer: bool = True, inner: bool = True) -> None:
    """Draw the chosen grid lines to the board.

    Args:
        surface (Surface): The surface being drawn to.
        outer (bool, optional): If the outer grid lines should be drawn. Defaults to True.
        inner (bool, optional): If the inner grid lines should be drawn. Defaults to True.
    """
    if (outer):
        _draw_outer_grid(surface)
        
    if (inner):
        _draw_inner_grid(surface)
    
def draw_rect(x: int, y: int, colour: tuple, surface: Surface) -> None:
    """Draw a rectangle to the board based on the grid size.

    Args:
        x (int): The grid based x position on the board to draw the rectangle.
        y (int): The grid based y position on the board to draw the rectangle.
        colour (tuple): The RGB colour of the rectangle.
        surface (Surface): The surface being drawn to.
    """
    x = x + pixel_to_grid_size(bd.BOARD_LEFT_BUFFER)
    y = y + pixel_to_grid_size(bd.BOARD_TOP_BUFFER) - bd.BOARD_HEIGHT_BUFFER
    
    draw.rect(surface, colour, Rect(x * Board.GRID_SIZE, y * Board.GRID_SIZE, Board.GRID_SIZE, Board.GRID_SIZE))