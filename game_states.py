from enum import Enum, auto

class GameStates(Enum):
    UPDATE_TIME = auto(),
    TAKE_INPUTS = auto(),
    RUN_LOGIC = auto(),
    DRAW_GAME = auto()