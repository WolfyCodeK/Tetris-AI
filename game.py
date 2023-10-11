import pygame
import board_definitions as bd

pygame.init()

pygame.display.set_caption('Tetris - Pygame')

window = pygame.display.set_mode((bd.SCR_WIDTH, bd.SCR_HEIGHT))
canvas = pygame.Surface((bd.SCR_WIDTH, bd.SCR_HEIGHT), pygame.SRCALPHA)

running = True

class zTetromino:
    def __init__(self, x: int, y: int) -> None:
        self.x = x
        self.y = y
        
    def drop(self):
        if (self.y < 19):
            self.y += 1

    def drawRect(self, x, y):
        x = x + bd.BOARD_LEFT_BUFFER / bd.SQUARE_SIZE
        y = y + bd.BOARD_TOP_BUFFER / bd.SQUARE_SIZE - (bd.DROP_HEIGHT / bd.SQUARE_SIZE)
        pygame.draw.rect(canvas, (255, 0, 0), pygame.Rect((x - 1) * bd.SQUARE_SIZE, (y - 1) * bd.SQUARE_SIZE, bd.SQUARE_SIZE, bd.SQUARE_SIZE))
    
    def draw(self):
        x = self.x
        y = self.y

        self.drawRect(x, y)
        self.drawRect(x + 1, y)
        self.drawRect(x + 1, y + 1)
        self.drawRect(x + 2, y + 1)

fps = 60
timer = pygame.time.Clock()
counter = 0

shape1 = zTetromino(4, 1)

font = pygame.font.Font('freesansbold.ttf', 32)
fpsString = font.render(str(fps), True, (255, 255, 255), (0, 0, 0))

while running:
    # Did the user click the window close button?
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
            
    window.fill(0)
            
    window.blit(canvas, (0, 0))        

    # Fill the background with white
    canvas.fill((0, 0, 0))

    if (counter > 60):
        fpsString = font.render(str(int(timer.get_fps())), True, (255, 255, 255), (0, 0, 0))
        shape1.drop()
        counter = 0

    # VERTICAL LINES
    for x in range(0, bd.BOARD_WIDTH, bd.SQUARE_SIZE):
        pygame.draw.line(canvas, (255, 255, 255, 100), (x + bd.BOARD_LEFT_BUFFER, bd.BOARD_TOP_BUFFER), (x + bd.BOARD_LEFT_BUFFER, bd.BOARD_HEIGHT + bd.BOARD_TOP_BUFFER))
    
    # HORIZONTAL LINES    
    for y in range(0, bd.BOARD_HEIGHT, bd.SQUARE_SIZE):
        pygame.draw.line(canvas, (255, 255, 255, 100), (bd.BOARD_LEFT_BUFFER, y + bd.BOARD_TOP_BUFFER), (bd.BOARD_WIDTH + bd.BOARD_LEFT_BUFFER, y + bd.BOARD_TOP_BUFFER))
        
    # Solid Board Outlines
    # TOP
    pygame.draw.line(canvas, (255, 255, 255, 255), bd.TOP_LEFT_BOARD_CORNER, bd.TOP_RIGHT_BOARD_CORNER)
    
    # BOTTOM
    pygame.draw.line(canvas, (255, 255, 255, 255), bd.BOTTOM_LEFT_BOARD_CORNER, bd.BOTTOM_RIGHT_BOARD_CORNER)
    
    # LEFT
    pygame.draw.line(canvas, (255, 255, 255, 255), bd.TOP_LEFT_BOARD_CORNER, bd.BOTTOM_LEFT_BOARD_CORNER)
    
    # RIGHT
    pygame.draw.line(canvas, (255, 255, 255, 255), bd.TOP_RIGHT_BOARD_CORNER, bd.BOTTOM_RIGHT_BOARD_CORNER)

    # Update the display
    pygame.display.flip()
    
    counter += 1
    
    timer.tick(fps)

    canvas.blit(fpsString, (bd.SCR_WIDTH - 50, bd.SQUARE_SIZE - 20))

# Done! Time to quit.
pygame.quit()