from pieces.i_piece import IPiece
from pieces.o_piece import OPiece
from pieces.piece import Piece
import board.board_definitions as bd
import board.board_utils as bu

class PieceHolder():
    def __init__(self) -> None:
        self.held_piece = None
        self.new_hold_available = True 
    
    def draw(self, surface):
        if (self.held_piece != None):
            
            for i in range(len(self.held_piece.shape)):
                x_adjust = bd.HELD_PIECE_X_POS
                y_adjust = bd.HELD_PIECE_Y_POS
                
                if (self.held_piece.pid == OPiece.PID):
                    x_adjust += 1
                if (self.held_piece.pid == IPiece.PID):
                    y_adjust -= 1
                    
                bu.draw_rect(self.held_piece.DEFAULT_SHAPE[i][0] + x_adjust, self.held_piece.DEFAULT_SHAPE[i][1] + y_adjust, self.held_piece.colour, surface)
    
    def hold_piece(self, current_piece: Piece) -> Piece:
        """Holds current piece and returns previously held piece if one is available.

        Args:
            current_piece (Piece): The piece to place into the hold position.

        Returns:
            bool: Returns the resulting current piece.
        """
        empty_hold = self.held_piece == None
        
        if (not empty_hold) and (self.new_hold_available):
            temp_piece = current_piece
            current_piece = self.held_piece
            self.held_piece = temp_piece
            
            self.new_hold_available = False
            
            self.held_piece.reset_piece()
            
        elif (empty_hold):
            self.held_piece = current_piece
            
            self.new_hold_available = False
            
            self.held_piece.reset_piece()
        else:
            current_piece = None
            
        return current_piece