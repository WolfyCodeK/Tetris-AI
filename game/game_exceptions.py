class PiecePlacementError(Exception):
    def __init__(self, x: int, y: int, id: chr, blocking_id: chr):
        self.x = x
        self.y = y
        self.piece = id
        self.message = f"Cannot place piece where blocking piece type already exists. Attempted to place {id} Piece at posistion: ({x}, {y}) but was already occupied by {blocking_id} Piece."
        super().__init__(self.message)
        
class RevertRotationError(Exception):
    def __init__(self, id: chr):
        self.id = id
        self.message = f"{id} - rotation has already been reverted, cannot revert rotation again without completing a new rotation."
        super().__init__(self.message)
        
class DrawException(Exception):
    def __init__(self):
        self.message = "Window cannot be drawn before running init_window()."
        super().__init__(self.message)
        
class ShapeStateMissing(Exception):
    def __init__(self, id: chr, shape) -> None:
        self.id = id
        self.shape = shape
        self.message = f"{shape}\n\n ^^^ is not a valid shape for {id}."
        super().__init__(self.message)