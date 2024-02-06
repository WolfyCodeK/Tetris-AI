from enum import Enum, auto


class GameStates(Enum):
    INIT_STATE = auto(),
    UPDATE_TIME = auto(),
    TAKE_INPUTS = auto(),
    RUN_LOGIC = auto(),
    DRAW_GAME = auto()