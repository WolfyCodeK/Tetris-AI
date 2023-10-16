class PiecePlacementError(Exception):
    def __init__(self, x, y, piece):
        self.x = x
        self.y = y
        self.piece = piece
        self.message = f"Cannot place piece where blocking piece type already exists. Attempted to place {piece} at posistion: ({x}, {y})"
        super().__init__(self.message)