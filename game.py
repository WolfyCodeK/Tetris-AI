import pygame

import board_utils as bu
import pieces
import piece_controller

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
window = pygame.display.set_mode((bu.SCR_WIDTH, bu.SCR_HEIGHT))
board_surface = pygame.Surface((bu.SCR_WIDTH, bu.SCR_HEIGHT), pygame.SRCALPHA)

# Set in-game fps
timer = pygame.time.Clock()
fps = 240
counter = 0

fps_string = font.render(str(fps), True, (0, 0, 0))

# Check if current piece is still in motion, or has locked in place

controller = piece_controller.PieceController(board_surface)

running = True

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
    
    # Clear window and draw window background       
    window.fill(0)     
    window.blit(board_surface, (0, 0))        
    board_surface.blit(background_image, (0, 0)) 
    
    # Draw board background
    pygame.draw.rect(board_surface, (0, 0, 0), pygame.Rect(bu.BOARD_LEFT_BUFFER, bu.BOARD_TOP_BUFFER, bu.BOARD_WIDTH, bu.BOARD_HEIGHT))
    
    # Draw finished pieces
    controller.draw_board_state()
    
    # Draw current piece
    controller.current_piece.draw()

    # Everying second update the fps counter and drop the tetris piece further
    if (counter > 60):
        fps_string = font.render(str(int(timer.get_fps())), True, (0, 0, 0))
        controller.drop_piece()
        counter = 0
        
    bu.draw_grid(board_surface)

    # Update window
    pygame.display.flip()
    
    counter += 1
    
    timer.tick(fps)

    board_surface.blit(fps_string, (bu.SCR_WIDTH - 50, bu.GRID_SIZE - 20))
    
    #print(controller.board_state)

pygame.quit()