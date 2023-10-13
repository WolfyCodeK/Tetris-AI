import pygame
import board_utils as bu
import piece_controller
import game_controller

# Pygame intial setup
pygame.init()
pygame.display.set_caption("Tetris - Pygame")

# Set window and surface sizes
window = pygame.display.set_mode((bu.SCR_WIDTH, bu.SCR_HEIGHT))
board_surface = pygame.Surface((bu.SCR_WIDTH, bu.SCR_HEIGHT), pygame.SRCALPHA)

# Set window icon
tetris_icon = pygame.image.load("res/tetris-icon.png")
pygame.display.set_icon(tetris_icon)

# Load images from resources
background_image = pygame.image.load("res/gradient_background_blue.jpg").convert()

# Create controllers for the game
p_controller = piece_controller.PieceController()
g_controller = game_controller.GameController(p_controller)

# Restrict how often player can input keys (temp solution)
register_input_delay = 0

running = True

while running:
    g_controller.update_delta_time()
    
    register_input_delay -= 1
    key = pygame.key.get_pressed()
    
    # Take player input
    if (key[pygame.K_RIGHT] == True) and (register_input_delay < 0):
        p_controller.shift_piece_horizontally(1)
        register_input_delay = 60
        
    if key[pygame.K_LEFT] == True and (register_input_delay < 0):
        p_controller.shift_piece_horizontally(-1)
        register_input_delay = 60
        
    if key[pygame.K_x] == True and (register_input_delay < 0):
        p_controller.rotate_piece(clockwise=True)
        register_input_delay = 60
    
    # Check pygame events
    for event in pygame.event.get():
        # Check if user has quit the window
        if event.type == pygame.QUIT:
            running = False
    
    # Run game logic
    g_controller.run_game_loop()
    
    # Clear window and draw window background       
    window.fill(0)     
    window.blit(board_surface, (0, 0))        

    # Create board background
    board_surface.fill((0, 0, 0, bu.BACKGROUND_ALPHA), pygame.Rect(bu.BOARD_LEFT_BUFFER, bu.BOARD_TOP_BUFFER, bu.BOARD_WIDTH, bu.BOARD_HEIGHT))
    
    # Draw all pieces
    g_controller.draw_pieces(board_surface)
    
    # Draw board background and window background
    window.blit(board_surface, (0, 0))
    board_surface.blit(background_image, (0, 0))
    
    # Draw board grids     
    bu.draw_grids(board_surface)
    
    # Draw fps counter
    board_surface.blit(g_controller.fps_string, (bu.SCR_WIDTH - 70, bu.GRID_SIZE - 20))

    # Update window
    pygame.display.flip()
    
    # Increment frames that have passed
    g_controller.frames += 1

pygame.quit()