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

tetris_icon = pygame.image.load("res/tetris-icon.png")
pygame.display.set_icon(tetris_icon)

# Load images from resources
background_image = pygame.image.load("res/gradient_background_blue.jpg").convert()

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
register_input_delay = 0

piece_deactive_delay_started = False
piece_deactive_delay = controller.PIECE_DEACTIVATE_DELAY

running = True

while running:
    delta_time = time.time() - previous_time
    previous_time = time.time()

    total_time += delta_time
    
    register_input_delay -= 1
    key = pygame.key.get_pressed()
    
    if (key[pygame.K_RIGHT] == True) and (register_input_delay < 0):
        controller.shift_piece_by_amount(1)
        register_input_delay = 60
        
    if key[pygame.K_LEFT] == True and (register_input_delay < 0):
        controller.shift_piece_by_amount(-1)
        register_input_delay = 60
    
    for event in pygame.event.get():
        # Check if user has quit the window
        if event.type == pygame.QUIT:
            running = False
    
    # Clear window and draw window background       
    window.fill(0)     
    window.blit(board_surface, (0, 0))        

    # Draw board background
    board_surface.fill((0, 0, 0, bu.BACKGROUND_ALPHA), pygame.Rect(bu.BOARD_LEFT_BUFFER, bu.BOARD_TOP_BUFFER, bu.BOARD_WIDTH, bu.BOARD_HEIGHT))
    
    # Draw finished pieces
    controller.draw_board_state(board_surface)
    
    # Draw current piece
    controller.draw_piece(board_surface)
    
    window.blit(board_surface, (0, 0))
    board_surface.blit(background_image, (0, 0))

    # Drop current piece and update fps counter
    if (total_time > controller.drop_time):
        if (fps_update_delay >= controller.drop_speed):
            fps_string = font.render(
                str(int(frames * controller.drop_speed)),
                True, 
                (0, 255, 0)
            )
            
            fps_update_delay = 0
        
        # Deactive piece after short delay if piece cannot be dropped
        if (not controller.drop_piece()):
            piece_deactive_delay_started = True
        else:
            piece_deactive_delay_started = False
            piece_deactive_delay = controller.PIECE_DEACTIVATE_DELAY
        
        # Start delay
        if (piece_deactive_delay_started):
            piece_deactive_delay -= 1
            
        # Deactivate piece once delay is done
        if (piece_deactive_delay < 0):
            controller.deactivate_piece()
            
            piece_deactive_delay_started = False
            piece_deactive_delay = controller.PIECE_DEACTIVATE_DELAY
        
        total_time = total_time - controller.drop_time
        frames = 0
        fps_update_delay += 1
        
    bu.draw_grids(board_surface)
    
    board_surface.blit(fps_string, (bu.SCR_WIDTH - 70, bu.GRID_SIZE - 20))

    # Update window
    pygame.display.flip()
    
    frames += 1

pygame.quit()