import utils.board_constants as bc
from controllers.board import Board
from .piece_holder import PieceHolder
from .piece_queue import PieceQueue
from pieces.piece import Piece
from game.game_exceptions import PiecePlacementError


class PieceManager():    
    def __init__(self) -> None:
        # Create game board
        self.board = Board()
        self.reset_board_and_pieces()
        
    def reset_board_and_pieces(self) -> None:
        self.board.reset_board_state()
        self.piece_queue = PieceQueue(Board.PIECE_LIST)
        self.piece_holder = PieceHolder()
        self.next_piece()
        
    def next_piece(self) -> None:
        self.current_piece = self.piece_queue.get_next_piece()
        
    def draw_board_pieces(self, surface):
        self.board.draw(surface)
            
    def draw_current_piece(self, surface):
        self.current_piece.draw(surface)
        
    def draw_ghost_pieces(self, surface):
        self.current_piece.draw_ghost(surface, self._calculate_max_drop_height())
        
    def draw_held_piece(self, surface):
        self.piece_holder.draw(surface)
        
    def draw_queued_pieces(self, surface):
        self.piece_queue.draw(surface)
        
    def gravity_drop_piece(self) -> bool:
        """Attempts to drop a tetramino piece down by one row.

        Returns:
            bool: True if the piece was successfully dropped down one row
        """
        DROP_AMOUNT = 1
        
        if (not self._piece_is_vertically_blocked(self.board.board_state, self.current_piece, DROP_AMOUNT)):
            self.current_piece.transform(0, DROP_AMOUNT)
            return True
        else:
            return False
        
    def _calculate_max_drop_height(self) -> int:
        piece_dropped = False
        drop_amount = 1
        
        while (not piece_dropped):
            if (not self._piece_is_vertically_blocked(self.board.board_state, self.current_piece, drop_amount)):
                drop_amount += 1
            else:
                piece_dropped = True
                
        return drop_amount - 1
        
    def hard_drop_piece(self) -> None:
        self.current_piece.transform(0, self._calculate_max_drop_height())
        
    def shift_piece_horizontally(self, x_move):
        if (not self._piece_is_horizontally_blocked(self.board.board_state, self.current_piece, x_move)):
            self.current_piece.transform(x_move, 0)
            
    def is_move_allowed(self, x: int, y: int) -> bool:
        if (x < 0) or (x >= bc.BOARD_WIDTH) or (y >= bc.BOARD_HEIGHT) or (self.board.board_state[y][x] in Board.PIECE_COLOUR_DICT.keys()):
            return False
        else:
            return True

    def rotate_piece(self, clockwise: bool) -> None:
        rotation_blocked = False
        
        piece = self.current_piece    
        piece.rotate(clockwise)
        
        # Attempt to place piece using basic rotation
        for i in range(len(piece.minos)):
            if (not self.is_move_allowed(piece.minos[i][0], piece.minos[i][1])):
                piece.revert_rotation()
                rotation_blocked = True
                break
        
        kick_found = False
        
        # If basic rotation didn't work, then attempt a kick
        if rotation_blocked:
            for i in range(len(piece.KICK_OPTIONS)):
                piece.rotate(clockwise)
                
                kick_priority = piece.get_kick_priority()
                kick_index = kick_priority[piece.rotation_state][i]
                    
                piece.kick(kick_index, clockwise)
                
                for j in range(len(piece.minos)):
                    if (not self.is_move_allowed(piece.minos[j][0], piece.minos[j][1])):
                        piece.revert_rotation()
                        piece.revert_kick()
                        kick_found = False
                        break
                    else:
                        kick_found = True  
                        
                if kick_found:
                    break
            
    def deactivate_piece(self) -> None:
        self._place_piece(self.board.board_state, self.current_piece)    
        
    def perform_line_clears(self) -> int:
        lines_cleared = 0
        
        for y in bc.BOARD_HEIGHT_RANGE_INCR:
            column_count = 0 
            
            for x in range(bc.BOARD_WIDTH):
                if self.board.board_state[y][x] in Board.PIECE_COLOUR_DICT.keys():
                    column_count += 1
                    
            if column_count == bc.BOARD_WIDTH:    
                # Move all lines down by one row
                for y2 in range(y, 1, -1):
                    self.board.board_state[y2] = self.board.board_state[y2 - 1]
                    
                lines_cleared += 1 
                    
        self.board.occupied_spaces -= lines_cleared * bc.BOARD_WIDTH               
    
        return lines_cleared
                    
    def _piece_is_vertically_blocked(self, board_state, piece: Piece, y_move) -> bool:
        blocked = False

        for i in range(len(piece.minos)):
            piece_y_pos = piece.minos[i][1] + y_move
            
            # Check the piece isnt going to hit the floor
            if (piece_y_pos >= bc.BOARD_HEIGHT):
                blocked = True
            else:
                # Check if piece is in the board
                if (piece.minos[i][1] + 1 >= 0):
                    pos_state = board_state[piece_y_pos][piece.minos[i][0]]
                    
                    # If there is a piece there then the position is blocked
                    if (pos_state != Board.EMPTY_PIECE_ID):  
                        blocked = True
                    
        return blocked
    
    def _piece_is_horizontally_blocked(self, board_state, piece: Piece, x_move) -> bool:
        blocked = False
        
        for i in range(len(piece.minos)):
            piece_pos = piece.minos[i][0] + x_move
            
            # Check for right input
            if (x_move > 0):
                if (piece_pos + x_move <= bc.BOARD_WIDTH):
                    if (board_state[piece.minos[i][1]][piece_pos] != Board.EMPTY_PIECE_ID):
                        blocked = True
                else:
                    blocked = True
            
            # Check for left input
            if (x_move < 0):
                if (piece_pos + x_move >= -1):
                    if (board_state[piece.minos[i][1]][piece_pos] != Board.EMPTY_PIECE_ID):
                        blocked = True
                else:
                    blocked = True          
            
        return blocked

    def _place_piece(self, board_state, piece: Piece):
        for i in range(len(piece.minos)):
            x = piece.minos[i][0]
            y = piece.minos[i][1]
            
            id = board_state[y][x]
            
            if (id == Board.EMPTY_PIECE_ID):
                board_state[piece.minos[i][1]][piece.minos[i][0]] = piece.id
            else:
                raise PiecePlacementError(x, y, piece.id, id)
            
        self.piece_holder.new_hold_available = True
        self.board.occupied_spaces += bc.PIECE_COMPONENTS
        
    def hold_piece(self):
        piece = self.piece_holder.hold_piece(self.current_piece)
        
        # The first time the hold has been used
        if piece == self.current_piece:
            self.next_piece()
        # No new hold was allowed
        elif piece == None:
            pass
        # Piece was switched with held piece
        else:
            self.current_piece = piece