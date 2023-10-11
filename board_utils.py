from pygame import draw, Rect

BOARD_ROWS = 20
BOARD_COLUMNS = 10

GRID_SIZE = 32
GRID_ALPHA = 50
GRID_OUTLINE_WIDTH = 3

DROP_HEIGHT = 1 * GRID_SIZE

BOARD_RIGHT_BUFFER = 6 * GRID_SIZE
BOARD_LEFT_BUFFER = 6 * GRID_SIZE
BOARD_TOP_BUFFER = 4 * GRID_SIZE
BOARD_BOTTOM_BUFFER = 4 * GRID_SIZE

BOARD_WIDTH = GRID_SIZE * BOARD_COLUMNS
BOARD_HEIGHT = GRID_SIZE * BOARD_ROWS

SCR_WIDTH = BOARD_WIDTH + BOARD_RIGHT_BUFFER + BOARD_LEFT_BUFFER
SCR_HEIGHT = BOARD_HEIGHT + BOARD_TOP_BUFFER + BOARD_BOTTOM_BUFFER

# Board Coordinate Definitions
TOP_LEFT_BOARD_CORNER = (BOARD_LEFT_BUFFER, BOARD_TOP_BUFFER)
TOP_RIGHT_BOARD_CORNER = (BOARD_LEFT_BUFFER + BOARD_WIDTH, BOARD_TOP_BUFFER)
BOTTOM_LEFT_BOARD_CORNER = (BOARD_LEFT_BUFFER, BOARD_TOP_BUFFER + BOARD_HEIGHT)
BOTTOM_RIGHT_BOARD_CORNER = (BOARD_LEFT_BUFFER + BOARD_WIDTH, BOARD_TOP_BUFFER + BOARD_HEIGHT)

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

def draw_grid(canvas):
    # VERTICAL LINES
    for x in range(0, BOARD_WIDTH, GRID_SIZE):
        draw.line(canvas, (255, 255, 255, GRID_ALPHA), (x + BOARD_LEFT_BUFFER, BOARD_TOP_BUFFER), (x + BOARD_LEFT_BUFFER, BOARD_HEIGHT + BOARD_TOP_BUFFER))
    
    # HORIZONTAL LINES    
    for y in range(0, BOARD_HEIGHT, GRID_SIZE):
        draw.line(canvas, (255, 255, 255, GRID_ALPHA), (BOARD_LEFT_BUFFER, y + BOARD_TOP_BUFFER), (BOARD_WIDTH + BOARD_LEFT_BUFFER, y + BOARD_TOP_BUFFER))
        
    # Solid Board Outlines
    # BOTTOM
    draw.line(canvas, (255, 255, 255, 255), BOTTOM_LEFT_BOARD_CORNER, BOTTOM_RIGHT_BOARD_CORNER, width=GRID_OUTLINE_WIDTH)
    
    # LEFT
    draw.line(canvas, (255, 255, 255, 255), TOP_LEFT_BOARD_CORNER, BOTTOM_LEFT_BOARD_CORNER, width=GRID_OUTLINE_WIDTH)
    
    # RIGHT
    draw.line(canvas, (255, 255, 255, 255), TOP_RIGHT_BOARD_CORNER, BOTTOM_RIGHT_BOARD_CORNER, width=GRID_OUTLINE_WIDTH)
    
def draw_rect(x, y, colour, board_surface):
    x = x + (BOARD_LEFT_BUFFER / GRID_SIZE)
    y = y + (BOARD_TOP_BUFFER / GRID_SIZE) - 1
    draw.rect(board_surface, colour, Rect(x * GRID_SIZE, y * GRID_SIZE, GRID_SIZE, GRID_SIZE))