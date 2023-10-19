from numpy import full
import utils.board_utils as bu

class Board():
    EMPTY_PIECE_PID = 'E'
    FLOOR_PIECE_PID = 'F'
    WALL_PIECE_PID = 'W'
    
    def __init__(self) -> None:
        self.board_state = self._init_board_state()
    
    def _init_board_state(self):
        # Initialise board state to be empty
        board_state = full(shape=(bu.BOARD_STATE_HEIGHT + bu.FLOOR_SIZE, bu.BOARD_STATE_WIDTH), fill_value=self.EMPTY_PIECE_PID)
            
        # Set wall pieces
        # Right wall
        for x in range(bu.BOARD_RIGHT_WALL, bu.BOARD_STATE_WIDTH):
            for y in range(len(board_state)):
                board_state[y][x] = self.WALL_PIECE_PID
            
        # Left wall
        for x in range(bu.BOARD_LEFT_WALL):
            for y in range(len(board_state)):
                board_state[y][x] = self.WALL_PIECE_PID
                
        # Set floor pieces
        for i in range(bu.FLOOR_SIZE):
            board_state[bu.BOARD_STATE_HEIGHT + i] = self.FLOOR_PIECE_PID
            
        return board_state