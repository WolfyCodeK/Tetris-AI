import random

import utils.game_settings as gs
import utils.board_utils as bu
from tetraminos.i_piece import IPiece
from tetraminos.o_piece import OPiece
from tetraminos.symmetrical_tetramino import (JPiece, LPiece, SPiece, TPiece, ZPiece)
from tetraminos.tetramino import Tetramino
from utils.custom_game_exceptions import PiecePlacementError
from piece_queue import PieceQueue
from board import Board


class PieceController():
    NONE_PIECE_TYPES = [Board.EMPTY_PIECE_PID, Board.FLOOR_PIECE_PID, Board.WALL_PIECE_PID]

    # All the available pieces to the piece controller
    PIECE_LIST = [ZPiece, SPiece, JPiece, LPiece, TPiece, IPiece, OPiece]
    
    PIECE_PID_LIST  = []
    for i in range(len(PIECE_LIST)):
                PIECE_PID_LIST.append(PIECE_LIST[i].PID)
    
    BLOCKING_PIECE_TYPES = PIECE_PID_LIST.copy()
    BLOCKING_PIECE_TYPES.append(Board.FLOOR_PIECE_PID)
    BLOCKING_PIECE_TYPES.append(Board.WALL_PIECE_PID)
    
    COLOUR_PID_DICT = {}
    for i in range(len(PIECE_LIST)):
        COLOUR_PID_DICT[PIECE_PID_LIST[i]] = PIECE_LIST[i].COLOUR
    
    def __init__(self) -> None:
        self.restart_board()
        
    def restart_board(self) -> None:
        self.board = Board()
        self.piece_queue = PieceQueue(self.PIECE_LIST)
        self.next_piece()
        
        self.held_piece = None
        self.new_hold_available = True    
        
    def next_piece(self) -> None:
        self.current_piece = self.piece_queue.get_next_piece()
        
    def draw_deactivated_pieces(self, surface):
        for y in range(len(self.board.board_state)):
            for x in range(len(self.board.board_state[0])):
                if (self.board.board_state[y][x] not in self.NONE_PIECE_TYPES):
                    bu.draw_rect(x, y, self.COLOUR_PID_DICT[self.board.board_state[y][x]], surface)
            
    def draw_current_piece(self, surface):
        self.current_piece.draw(surface)
        
    def draw_ghost_pieces(self, surface):
        self.current_piece.draw_ghost(surface, self._calculate_max_drop_height())
        
    def draw_held_piece(self, surface):
        if (self.held_piece != None):
            self.held_piece.reset_shape()
            
            for i in range(len(self.held_piece.shape)):
                x_adjust = bu.HELD_PIECE_X_POS
                y_adjust = bu.HELD_PIECE_Y_POS
                
                if (self.held_piece.pid == OPiece.PID):
                    x_adjust += 1
                if (self.held_piece.pid == IPiece.PID):
                    y_adjust -= 1
                    
                bu.draw_rect(self.held_piece.shape[i][0] + x_adjust, self.held_piece.shape[i][1] + y_adjust, self.held_piece.colour, surface)
                
    def draw_queued_pieces(self, surface):
        for i in range(gs.NUM_OF_QUEUE_TO_SHOW):
            # Get piece in queue
            piece = self.piece_queue.queue[i]
            piece.reset_shape()
            
            for j in range(len(piece.shape)):
                x_adjust = bu.QUEUED_PIECES_X_POS
                y_adjust = bu.QUEUED_PIECES_Y_POS
                
                if (piece.pid == OPiece.PID):
                    x_adjust += 1
                if (piece.pid == IPiece.PID):
                    y_adjust -= 1
                    
                bu.draw_rect(piece.shape[j][0] + x_adjust, piece.shape[j][1] + y_adjust + (i * bu.QUEUED_PIECES_VERTICAL_SPACING), piece.colour, surface)
        
    def gravity_drop_piece(self) -> bool:
        """Attempts to drop a tetramino piece down by one row.

        Returns:
            bool: True if the piece was successfully dropped down one row
        """
        if (not self._piece_is_vertically_blocked(self.board.board_state, self.current_piece, 1)):
            self.current_piece.set_y_pos(self.current_piece.y_pos + 1)
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
        self.current_piece.set_y_pos(self.current_piece.y_pos + self._calculate_max_drop_height())
        
    def shift_piece_horizontally(self, x_move):
        if (not self._piece_is_horizontally_blocked(self.board.board_state, self.current_piece, x_move)):
            self.current_piece.set_x_pos(self.current_piece.x_pos + x_move)
            
    def rotate_piece(self, clockwise: bool) -> None:
        piece = self.current_piece    
        
        piece.rotate(clockwise)
        
        blocked = False
        
        # Attempt to place piece using basic rotation
        for i in range(len(piece.minos)):
            if (self.board.board_state[piece.minos[i][1]][piece.minos[i][0]] in self.BLOCKING_PIECE_TYPES):
                piece.revert_rotation()
                blocked = True
                break
        
        kick_found = False
        
        # If basic rotation didn't work, then attempt a kick
        if blocked:
            for i in range(len(piece.kick_options)):
                piece.rotate(clockwise)
                
                kick_priority = piece.get_kick_priority()
                kick_index = kick_priority[piece.rotation_state][i]
                    
                piece.save_previous_pos()
                piece.kick(kick_index, clockwise)
                
                for j in range(len(piece.minos)):
                    if (self.board.board_state[piece.minos[j][1]][piece.minos[j][0]] in self.BLOCKING_PIECE_TYPES):
                        piece.revert_rotation()
                        piece.revert_kick()
                        kick_found = False
                        break
                    else:
                        kick_found = True  
                        
                if kick_found:
                    break
            
    def deactivate_piece(self) -> None:
        self.current_piece.active = False
        self._place_piece(self.board.board_state, self.current_piece)    
        
    def perform_line_clears(self) -> int:
        lines_cleared = 0
        
        for y in range(len(self.board.board_state)):
            column_count = 0 
            
            for x in range(bu.BOARD_STATE_WIDTH):
                if self.board.board_state[y][x] not in self.NONE_PIECE_TYPES:
                    column_count += 1
                    
            if column_count >= bu.BOARD_COLUMNS:    
                lines_cleared += 1   
                
                for y2 in range(y, 1, -1):
                    self.board.board_state[y2] = self.board.board_state[y2 - 1]
                    
        return lines_cleared
                    
    def check_game_over(self) -> bool:
        for y in range(bu.BOARD_STATE_HEIGHT_BUFFER):
            if any(pid in self.PIECE_PID_LIST for pid in self.board.board_state[y].tolist()):
                return True

    def _piece_is_vertically_blocked(self, board_state, piece: Tetramino, y_move) -> bool:
        blocked = False

        for i in range(len(piece.minos)):
            piece_pos = piece.minos[i][1] + y_move
            
            # Check the piece isnt going to hit the floor
            if (piece_pos != Board.FLOOR_PIECE_PID):
                # Check if piece is in the board
                if (piece.minos[i][1] + 1 >= 0):
                    pos_state = board_state[piece_pos][piece.minos[i][0]]
                    
                    # If there is a piece there then the position is blocked
                    if (pos_state != Board.EMPTY_PIECE_PID):  
                        blocked = True
            else:
                blocked = True
                    
        return blocked
    
    def _piece_is_horizontally_blocked(self, board_state, piece: Tetramino, x_move) -> bool:
        blocked = False
        
        for i in range(len(piece.minos)):
            piece_pos = piece.minos[i][0] + x_move
            
            # Check for right input
            if (x_move > 0):
                if (piece_pos + x_move <= bu.BOARD_RIGHT_WALL):
                    if (board_state[piece.minos[i][1]][piece_pos] != Board.EMPTY_PIECE_PID):
                        blocked = True
                else:
                    blocked = True
            
            # Check for left input
            if (x_move < 0):
                if (piece_pos + x_move >= bu.BOARD_LEFT_WALL - 1):
                    if (board_state[piece.minos[i][1]][piece_pos] != Board.EMPTY_PIECE_PID):
                        blocked = True
                else:
                    blocked = True          
            
        return blocked

    def _place_piece(self, board_state, piece: Tetramino):
        for i in range(len(piece.minos)):
            y = piece.minos[i][1]
            x = piece.minos[i][0]
            
            if (board_state[y][x] not in self.BLOCKING_PIECE_TYPES):
                board_state[piece.minos[i][1]][piece.minos[i][0]] = piece.pid
            else:
                raise PiecePlacementError(x, y, piece.pid)
            
        self.new_hold_available = True
        
    def hold_piece(self):
        if (self.held_piece != None) and (self.new_hold_available):
            temp_piece = self.current_piece
            self.current_piece = self.held_piece
            self.held_piece = temp_piece
            
            self.new_hold_available = False
            
            self.held_piece.reset_pos()
            
        elif (self.held_piece == None):
            self.held_piece = self.current_piece
            self.next_piece()
            
            self.new_hold_available = False
            
            self.held_piece.reset_pos()