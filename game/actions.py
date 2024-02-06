from enum import IntEnum

class Actions(IntEnum):
    MOVE_LEFT = 0,
    MOVE_RIGHT = 1,
    ROTATE_CLOCKWISE = 2,
    ROTATE_ANTICLOCKWISE = 3,
    SOFT_DROP = 4,
    HARD_DROP = 5,
    HOLD_PIECE = 6
