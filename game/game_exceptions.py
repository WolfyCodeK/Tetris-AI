class PiecePlacementError(Exception):
    def __init__(self, x: int, y: int, pid: chr, blocking_pid: chr):
        self.x = x
        self.y = y
        self.piece = pid
        self.message = f"Cannot place piece where blocking piece type already exists. Attempted to place {pid} Piece at posistion: ({x}, {y}) but was already occupied by {blocking_pid} Piece."
        super().__init__(self.message)
        
class RevertRotationError(Exception):
    def __init__(self, pid: chr):
        self.pid = pid
        self.message = f"{pid} - rotation has already been reverted, cannot revert rotation again without completing a new rotation."
        super().__init__(self.message)