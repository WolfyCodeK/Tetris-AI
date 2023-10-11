import tetramino
import numpy as np

class PieceController():
    def __init__(self, board_state: np.ndarray) -> None:
        self.board_state = board_state
        
    def add_piece_to_board(piece: tetramino.Tetramino) -> None:
        #piece.
        pass
        
    def deactivate_piece(self, piece: tetramino.Tetramino) -> None:
        piece.active = False
        self.controller.add_piece_to_board(piece)