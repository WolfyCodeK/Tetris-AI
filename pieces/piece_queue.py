import random
from pieces.piece_type_id import PieceTypeID
import utils.board_constants as bc
import utils.window_utils as win_utils
from game.game_settings import GameSettings
from pieces.pieces import IPiece, OPiece
from pieces.piece import Piece

class PieceQueue():
    def __init__(self, piece_list: list[Piece]) -> None:
        self.PIECE_LIST = piece_list
        self.NUM_OF_PIECES = len(self.PIECE_LIST)
        self.LIST_OF_PIECE_NUMBERS = list(range(self.NUM_OF_PIECES))
        self.NONE_STARTING_PIECE_IDS = [(int(pid) - 1) for pid in [PieceTypeID.O_PIECE, PieceTypeID.Z_PIECE, PieceTypeID.S_PIECE]]
        """
        (int(pid) - 1), this is because LIST_OF_PIECE_NUMBERS starts from 0, whereas the piece ids start from 1
        """ 
        
        self.NUM_OF_PIECES_TO_SHOW = GameSettings.num_of_queue_to_show
        self.first_bag_numbers = self._create_random_piece_bag()
        self.second_bag_numbers = self._create_random_piece_bag()
        
        self.queue = self._init_piece_queue()
    
    def draw(self, surface):
        for i in range(self.NUM_OF_PIECES_TO_SHOW):
            # Get piece in queue
            piece = self.queue[i]
            
            for j in range(len(piece.shape)):
                x_adjust = bc.QUEUED_PIECES_X_POS
                y_adjust = bc.QUEUED_PIECES_Y_POS
                
                if (piece.id == OPiece.ID):
                    x_adjust += 1
                if (piece.id == IPiece.ID):
                    y_adjust -= 1
                    
                win_utils.draw_rect(piece.DEFAULT_SHAPE[j][0] + x_adjust, piece.DEFAULT_SHAPE[j][1] + y_adjust + (i * bc.QUEUED_PIECES_VERTICAL_SPACING), piece.colour, surface)
    
    def _init_piece_queue(self) -> list[Piece]:
        queue = []
        
        for i in range(self.NUM_OF_PIECES):
            queue.append(self.PIECE_LIST[self.first_bag_numbers[i]]())
        
        return queue
    
    def _create_random_piece_bag(self) -> list[int]:
        bag = self.LIST_OF_PIECE_NUMBERS.copy()
        random.seed(GameSettings.seed)
        random.shuffle(bag)
        
        while any([bag[0] == int(pid) for pid in self.NONE_STARTING_PIECE_IDS]) and any([bag[1] == int(pid) for pid in self.NONE_STARTING_PIECE_IDS]):
            random.shuffle(bag)
        
        return bag
    
    def get_next_piece(self) -> Piece:
        piece = self.queue.pop(0)
        self._add_piece_to_queue()
        
        return piece
    
    def get_truncated_piece_queue(self, first_n_pieces) -> list:
        queue_id_list = [int(piece.id) for piece in self.queue]
        
        for _ in range(self.NUM_OF_PIECES - first_n_pieces):
            queue_id_list.pop()
            
        return queue_id_list
        
    def _add_piece_to_queue(self) -> None:
        if (len(self.second_bag_numbers) <= 0):
            self.second_bag_numbers = self._create_random_piece_bag()
            
        piece_num = self.first_bag_numbers.pop(0)
        self.first_bag_numbers.append(self.second_bag_numbers.pop(0))

        self.queue.append(self.PIECE_LIST[piece_num]())