class PiecePlacementError(Exception):
    def __init__(self, x, y, piece, blocking):
        self.x = x
        self.y = y
        self.piece = piece
        self.message = f"Cannot place piece where blocking piece type already exists. Attempted to place {piece} Piece at posistion: ({x}, {y}) but was already occupied by {blocking} Piece."
        super().__init__(self.message)