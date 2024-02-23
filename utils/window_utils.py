from game.game_settings import GameSettings
from pygame import Rect, Surface, draw
import utils.board_constants as bc
    
def get_grid_size():
    return GameSettings.screen_size * 2      
    
def get_board_buffers():
    """Gets all the buffers for the edges of the board.

    Returns:
        tuple(int, int, int, int): right buffer, left buffer, top buffer, bottom buffer
    """
    # Board drawing helper value
    BOARD_RIGHT_BUFFER = 7 * get_grid_size()
    BOARD_LEFT_BUFFER = 7 * get_grid_size()
    BOARD_TOP_BUFFER = 4 * get_grid_size()
    BOARD_BOTTOM_BUFFER = 4 * get_grid_size()
    
    return BOARD_RIGHT_BUFFER, BOARD_LEFT_BUFFER, BOARD_TOP_BUFFER, BOARD_BOTTOM_BUFFER

def get_board_sizes():
    # Base board size in pixel
    BOARD_PIXEL_WIDTH = get_grid_size() * bc.BOARD_COLUMNS
    BOARD_PIXEL_HEIGHT = get_grid_size() * bc.BOARD_ROWS
    
    return BOARD_PIXEL_WIDTH, BOARD_PIXEL_HEIGHT

def get_screen_sizes():
    width, height = get_board_sizes()
    right_buf, left_buf, top_buf, bottom_buf = get_board_buffers()
    
    # Screen size
    SCR_WIDTH = width + right_buf + left_buf
    SCR_HEIGHT = height + top_buf + bottom_buf
    
    return SCR_WIDTH, SCR_HEIGHT

def get_board_corner_coordiantes():
    """Gets all the corner coordiantes of the board.

    Returns:
        tuple(int, int, int, int): top left corner, top right corner, bottom left corner, bottom right corner
    """
    _, left_buf, top_buf, _ = get_board_buffers()
    width, height = get_board_sizes()
    
    # Board Coordinate Definitions
    TOP_LEFT_BOARD_CORNER = (left_buf - 2, top_buf)
    TOP_RIGHT_BOARD_CORNER = (left_buf + width + 1, top_buf)
    BOTTOM_LEFT_BOARD_CORNER = (left_buf - 2, top_buf + height + 1)
    BOTTOM_RIGHT_BOARD_CORNER = (left_buf + width + 1, top_buf + height + 1)
    
    return TOP_LEFT_BOARD_CORNER, TOP_RIGHT_BOARD_CORNER, BOTTOM_LEFT_BOARD_CORNER, BOTTOM_RIGHT_BOARD_CORNER

def pixel_to_grid_size(x: int) -> int:
    """Converts pixel size to grid size e.g. 160 pixels to 5 squares, where grid size = 32.

    Args:
        x (int): Any length of pixels

    Returns:
        int: That length divided by the grid size
    """    
    return x / get_grid_size()

def grid_to_pixel_length(x: int) -> int:
    """Converts grid size to pixel size e.g. 5 squares to 160 pixels, where grid size = 32.

    Args:
        x (int): Any number of squares

    Returns:
        int: That length multiplied by the grid size
    """
    return x * get_grid_size()

def _draw_outer_grid(surface: Surface) -> None:
    """Draw the thick outline of the boards grid.

    Args:
        surface (Surface): The surface being drawn to.
    """
    top_left, top_right, bottom_left, bottom_right = get_board_corner_coordiantes()
    
    # BOTTOM
    draw.line(surface, (255, 255, 255, bc.OUTER_GRID_ALPHA), bottom_left, bottom_right, width=bc.GRID_OUTLINE_WIDTH)
    
    # LEFT
    draw.line(surface, (255, 255, 255, bc.OUTER_GRID_ALPHA), top_left, bottom_left, width=bc.GRID_OUTLINE_WIDTH)
    
    # RIGHT
    draw.line(surface, (255, 255, 255, bc.OUTER_GRID_ALPHA), top_right, bottom_right, width=bc.GRID_OUTLINE_WIDTH)

def _draw_inner_grid(surface: Surface) -> None:
    """Draw the thin inner lines of the boards grid.

    Args:
        surface (Surface): The surface being drawn to.
    """
    width, height = get_board_sizes()
    _, left_buf, top_buf, _ = get_board_buffers()
    
    # VERTICAL LINES
    for x in range(1 * get_grid_size(), width, get_grid_size()):
        draw.line(
            surface, 
            (255, 255, 255, bc.INNER_GRID_ALPHA), 
            (x + left_buf, top_buf + 2), 
            (x + left_buf, height + top_buf - 1)
        )
    
    # HORIZONTAL LINES    
    for y in range(1 * get_grid_size(), height, get_grid_size()):
        draw.line(
            surface, 
            (255, 255, 255, bc.INNER_GRID_ALPHA), 
            (left_buf, y + top_buf), 
            (width + left_buf - 1, y + top_buf)
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
    _, left_buf, top_buf, _ = get_board_buffers()
    
    x = x + pixel_to_grid_size(left_buf)
    y = y + pixel_to_grid_size(top_buf) - bc.BOARD_HEIGHT_BUFFER
    
    draw.rect(surface, colour, Rect(x * get_grid_size(), y * get_grid_size(), get_grid_size(), get_grid_size()))
    