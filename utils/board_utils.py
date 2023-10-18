from pygame import Rect, Surface, draw
import game_settings as gs

BOARD_ROWS = 20
BOARD_COLUMNS = 10

GRID_SIZE = gs.GAME_SIZE * 2
INNER_GRID_ALPHA = 40
OUTER_GRID_ALPHA = 200
BACKGROUND_ALPHA = 150
GRID_OUTLINE_WIDTH = 3

MAX_PIECE_LENGTH = 2
DROP_HEIGHT = 2
FLOOR_SIZE = 2

HELD_PIECE_X_POS = -2
HELD_PIECE_Y_POS = 6

QUEUED_PIECES_X_POS = 15
QUEUED_PIECES_Y_POS = 6
QUEUED_PIECES_VERTICAL_SPACING = 3

BOARD_STATE_HEIGHT_BUFFER = DROP_HEIGHT + MAX_PIECE_LENGTH
BOARD_STATE_WIDTH_BUFFER = MAX_PIECE_LENGTH

BOARD_STATE_HEIGHT = BOARD_ROWS + BOARD_STATE_HEIGHT_BUFFER
BOARD_STATE_WIDTH = BOARD_COLUMNS + BOARD_STATE_WIDTH_BUFFER * 2

BOARD_LEFT_WALL = BOARD_STATE_WIDTH_BUFFER
BOARD_RIGHT_WALL = BOARD_STATE_WIDTH_BUFFER + BOARD_COLUMNS

BOARD_RIGHT_BUFFER = 7 * GRID_SIZE
BOARD_LEFT_BUFFER = 7 * GRID_SIZE
BOARD_TOP_BUFFER = 4 * GRID_SIZE
BOARD_BOTTOM_BUFFER = 4 * GRID_SIZE

BOARD_WIDTH = GRID_SIZE * BOARD_COLUMNS
BOARD_HEIGHT = GRID_SIZE * BOARD_ROWS

SCR_WIDTH = BOARD_WIDTH + BOARD_RIGHT_BUFFER + BOARD_LEFT_BUFFER
SCR_HEIGHT = BOARD_HEIGHT + BOARD_TOP_BUFFER + BOARD_BOTTOM_BUFFER

# Board Coordinate Definitions
TOP_LEFT_BOARD_CORNER = (BOARD_LEFT_BUFFER - 2, BOARD_TOP_BUFFER)
TOP_RIGHT_BOARD_CORNER = (BOARD_LEFT_BUFFER + BOARD_WIDTH + 1, BOARD_TOP_BUFFER)
BOTTOM_LEFT_BOARD_CORNER = (BOARD_LEFT_BUFFER - 2, BOARD_TOP_BUFFER + BOARD_HEIGHT + 1)
BOTTOM_RIGHT_BOARD_CORNER = (BOARD_LEFT_BUFFER + BOARD_WIDTH + 1, BOARD_TOP_BUFFER + BOARD_HEIGHT + 1)

def pixel_to_grid_size(x: int) -> int:
    """Converts pixel size to grid size e.g. 160 pixels to 5 squares, where grid size = 32.

    Args:
        x (int): Any length of pixels

    Returns:
        int: That length divided by the grid size
    """    
    return x / GRID_SIZE

def grid_to_pixel_length(x: int) -> int:
    """Converts grid size to pixel size e.g. 5 squares to 160 pixels, where grid size = 32.

    Args:
        x (int): Any number of squares

    Returns:
        int: That length multiplied by the grid size
    """
    return x * GRID_SIZE

def _draw_outer_grid(surface: Surface):
    """Draw the thick outline of the boards grid.

    Args:
        surface (Surface): The surface being drawn to.
    """
    # BOTTOM
    draw.line(surface, (255, 255, 255, OUTER_GRID_ALPHA), BOTTOM_LEFT_BOARD_CORNER, BOTTOM_RIGHT_BOARD_CORNER, width=GRID_OUTLINE_WIDTH)
    
    # LEFT
    draw.line(surface, (255, 255, 255, OUTER_GRID_ALPHA), TOP_LEFT_BOARD_CORNER, BOTTOM_LEFT_BOARD_CORNER, width=GRID_OUTLINE_WIDTH)
    
    # RIGHT
    draw.line(surface, (255, 255, 255, OUTER_GRID_ALPHA), TOP_RIGHT_BOARD_CORNER, BOTTOM_RIGHT_BOARD_CORNER, width=GRID_OUTLINE_WIDTH)
    
def _draw_inner_grid(surface: Surface):
    """Draw the thin inner lines of the boards grid.

    Args:
        surface (Surface): The surface being drawn to.
    """
    # VERTICAL LINES
    for x in range(1 * GRID_SIZE, BOARD_WIDTH, GRID_SIZE):
        draw.line(
            surface, 
            (255, 255, 255, INNER_GRID_ALPHA), 
            (x + BOARD_LEFT_BUFFER, BOARD_TOP_BUFFER + 2), 
            (x + BOARD_LEFT_BUFFER, BOARD_HEIGHT + BOARD_TOP_BUFFER - 1)
        )
    
    # HORIZONTAL LINES    
    for y in range(1 * GRID_SIZE, BOARD_HEIGHT, GRID_SIZE):
        draw.line(
            surface, 
            (255, 255, 255, INNER_GRID_ALPHA), 
            (BOARD_LEFT_BUFFER, y + BOARD_TOP_BUFFER), 
            (BOARD_WIDTH + BOARD_LEFT_BUFFER - 1, y + BOARD_TOP_BUFFER)
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
    
def draw_rect(x: int, y: int, colour: tuple, surface: Surface):
    """Draw a rectangle to the board based on the grid size.

    Args:
        x (int): The grid based x position on the board to draw the rectangle.
        y (int): The grid based y position on the board to draw the rectangle.
        colour (tuple): The RGB colour of the rectangle.
        surface (Surface): The surface being drawn to.
    """
    x = x + pixel_to_grid_size(BOARD_LEFT_BUFFER) - BOARD_STATE_WIDTH_BUFFER
    y = y + pixel_to_grid_size(BOARD_TOP_BUFFER) - BOARD_STATE_HEIGHT_BUFFER
    draw.rect(surface, colour, Rect(x * GRID_SIZE, y * GRID_SIZE, GRID_SIZE, GRID_SIZE))