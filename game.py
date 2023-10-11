import pygame

import board_definitions as bd
import tetraminos

pygame.init()

pygame.display.set_caption('Tetris - Pygame')

programIcon = pygame.image.load('res\Tetris_S.svg.png')

pygame.display.set_icon(programIcon)

window = pygame.display.set_mode((bd.SCR_WIDTH, bd.SCR_HEIGHT))
canvas = pygame.Surface((bd.SCR_WIDTH, bd.SCR_HEIGHT), pygame.SRCALPHA)

running = True

fps = 60
timer = pygame.time.Clock()
counter = 0

shape1 = tetraminos.zTetromino(4, 1, canvas)

font = pygame.font.Font('freesansbold.ttf', 32)
fpsString = font.render(str(fps), True, (0, 0, 0))

while running:
    # Did the user click the window close button?
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
            
    window.fill(0)
            
    window.blit(canvas, (0, 0))        
    canvas.fill((100, 100, 100))
    
    pygame.draw.rect(canvas, (0, 0, 0), pygame.Rect(bd.BOARD_LEFT_BUFFER, bd.BOARD_TOP_BUFFER, bd.BOARD_WIDTH, bd.BOARD_HEIGHT))

    shape1.draw()

    if (counter > 60):
        fpsString = font.render(str(int(timer.get_fps())), True, (0, 0, 0))
        shape1.drop()
        counter = 0

    # VERTICAL LINES
    for x in range(0, bd.BOARD_WIDTH, bd.GRID_SIZE):
        pygame.draw.line(canvas, (255, 255, 255, bd.GRID_ALPHA), (x + bd.BOARD_LEFT_BUFFER, bd.BOARD_TOP_BUFFER), (x + bd.BOARD_LEFT_BUFFER, bd.BOARD_HEIGHT + bd.BOARD_TOP_BUFFER))
    
    # HORIZONTAL LINES    
    for y in range(0, bd.BOARD_HEIGHT, bd.GRID_SIZE):
        pygame.draw.line(canvas, (255, 255, 255, bd.GRID_ALPHA), (bd.BOARD_LEFT_BUFFER, y + bd.BOARD_TOP_BUFFER), (bd.BOARD_WIDTH + bd.BOARD_LEFT_BUFFER, y + bd.BOARD_TOP_BUFFER))
        
    # Solid Board Outlines
    
    # BOTTOM
    pygame.draw.line(canvas, (255, 255, 255, 255), bd.BOTTOM_LEFT_BOARD_CORNER, bd.BOTTOM_RIGHT_BOARD_CORNER, width=bd.GRID_OUTLINE_WIDTH)
    
    # LEFT
    pygame.draw.line(canvas, (255, 255, 255, 255), bd.TOP_LEFT_BOARD_CORNER, bd.BOTTOM_LEFT_BOARD_CORNER, width=bd.GRID_OUTLINE_WIDTH)
    
    # RIGHT
    pygame.draw.line(canvas, (255, 255, 255, 255), bd.TOP_RIGHT_BOARD_CORNER, bd.BOTTOM_RIGHT_BOARD_CORNER, width=bd.GRID_OUTLINE_WIDTH)

    # Update the display
    pygame.display.flip()
    
    counter += 1
    
    timer.tick(fps)

    canvas.blit(fpsString, (bd.SCR_WIDTH - 50, bd.GRID_SIZE - 20))

# Done! Time to quit.
pygame.quit()