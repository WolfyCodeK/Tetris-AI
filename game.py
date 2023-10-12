import time
import pygame

import board_utils as bu
import piece_controller

# Pygame intial setup
pygame.init()
pygame.display.set_caption("Tetris - Pygame")

# Set window and surface sizes
window = pygame.display.set_mode((bu.SCR_WIDTH, bu.SCR_HEIGHT))
board_surface = pygame.Surface((bu.SCR_WIDTH, bu.SCR_HEIGHT), pygame.SRCALPHA)

# Load images from resources
tetris_icon = pygame.image.load("res/tetris-icon.png").convert()
background_image = pygame.image.load("res/gradient_background_blue.jpg").convert()

pygame.display.set_icon(tetris_icon)

# Create fonts
font = pygame.font.Font("freesansbold.ttf", 32)

# Set in-game fps
fps_string = font.render(str("- - -"), True, (0, 255, 0))

controller = piece_controller.PieceController()
vertical_movements = 0

previous_time = time.time()
total_time = 0
frames = 0
fps_update_delay = 0

running = True

while running:
    delta_time = time.time() - previous_time
    previous_time = time.time()

    total_time += delta_time
    
    for event in pygame.event.get():
        # Check if user has quit the window
        if event.type == pygame.QUIT:
            running = False
    
    # Clear window and draw window background       
    window.fill(0)     
    window.blit(board_surface, (0, 0))        
    board_surface.blit(background_image, (0, 0)) 
    
    # Draw board background
    pygame.draw.rect(board_surface, (0, 0, 0), pygame.Rect(bu.BOARD_LEFT_BUFFER, bu.BOARD_TOP_BUFFER, bu.BOARD_WIDTH, bu.BOARD_HEIGHT))
    
    # Draw finished pieces
    controller.draw_board_state(board_surface)
    
    # Draw current piece
    controller.draw_piece(board_surface)

    # Drop current piece and update fps counter
    if (total_time > controller.drop_time):
        if (fps_update_delay >= controller.drop_speed):
            fps_string = font.render(
                str(int(frames * controller.drop_speed)),
                True, 
                (0, 255, 0)
            )
            
            fps_update_delay = 0

        controller.drop_piece() 

        total_time = total_time - controller.drop_time
        frames = 0
        fps_update_delay += 1
        
    bu.draw_grid(board_surface)
    
    board_surface.blit(fps_string, (bu.SCR_WIDTH - 70, bu.GRID_SIZE - 20))

    # Update window
    pygame.display.flip()
    
    frames += 1

pygame.quit()