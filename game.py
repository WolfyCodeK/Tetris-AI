import numpy as np
import pygame

import board_utils as bd
import pieces

# Load images from resources
tetris_icon = pygame.image.load('res/tetris-icon.png')
background_image = pygame.image.load('res/gradient_background_blue.jpg')

# Pygame intial setup
pygame.init()
pygame.display.set_caption('Tetris - Pygame')
pygame.display.set_icon(tetris_icon)

# Create fonts
font = pygame.font.Font('freesansbold.ttf', 32)

# Set window and surface sizes
window = pygame.display.set_mode((bd.SCR_WIDTH, bd.SCR_HEIGHT))
main_surface = pygame.Surface((bd.SCR_WIDTH, bd.SCR_HEIGHT), pygame.SRCALPHA)

# Set in-game fps
timer = pygame.time.Clock()
fps = 60
counter = 0

fps_string = font.render(str(fps), True, (0, 0, 0))

# Check if current piece is still in motion, or has locked in place
current_piece = pieces.ZPiece(main_surface)
is_piece_active = current_piece.active

# Initialise board state to be empty
boardState = np.zeros(shape=(bd.BOARD_ROWS, bd.BOARD_COLUMNS))

running = True

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
    
    # Clear window and draw window background       
    window.fill(0)     
    window.blit(main_surface, (0, 0))        
    main_surface.blit(background_image, (0, 0)) 
    
    # Draw board background
    pygame.draw.rect(main_surface, (0, 0, 0), pygame.Rect(bd.BOARD_LEFT_BUFFER, bd.BOARD_TOP_BUFFER, bd.BOARD_WIDTH, bd.BOARD_HEIGHT))

    current_piece.draw()

    # Everying second update the fps counter and drop the tetris piece further
    if (counter > 60):
        fps_string = font.render(str(int(timer.get_fps())), True, (0, 0, 0))
        #shape_1.drop()
        current_piece.y = current_piece.y + 1
        counter = 0
        
    bd.draw_grid(main_surface)

    # Update window
    pygame.display.flip()
    
    counter += 1
    
    timer.tick(fps)

    main_surface.blit(fps_string, (bd.SCR_WIDTH - 50, bd.GRID_SIZE - 20))

pygame.quit()