from random import shuffle
from tetraminos.tetramino import Tetramino

class PieceQueue():
    def __init__(self, piece_list: list[Tetramino]) -> None:
        self.PIECE_LIST = piece_list
        self.NUM_OF_PIECES = len(self.PIECE_LIST)
        self.LIST_OF_PIECE_NUMBERS = list(range(0, self.NUM_OF_PIECES))
        
        self.first_bag_numbers = self._create_random_piece_bag()
        self.second_bag_numbers = self._create_random_piece_bag()
        
        self.queue = self._init_piece_queue()
    
    def _init_piece_queue(self) -> list[Tetramino]:
        queue = []
        
        for i in range(self.NUM_OF_PIECES):
            queue.append(self.PIECE_LIST[self.first_bag_numbers[i]]())
        
        return queue
    
    def _create_random_piece_bag(self) -> list[int]:
        bag = self.LIST_OF_PIECE_NUMBERS.copy()
        shuffle(bag)
        
        return bag
    
    def get_next_piece(self) -> Tetramino:
        piece = self.queue.pop(0)
        self._add_piece_to_queue()
        
        return piece
        
    def _add_piece_to_queue(self) -> None:
        if (len(self.second_bag_numbers) <= 0):
            self.second_bag_numbers = self._create_random_piece_bag()
            
        piece_num = self.first_bag_numbers.pop(0)
        self.first_bag_numbers.append(self.second_bag_numbers.pop(0))

        self.queue.append(self.PIECE_LIST[piece_num]())