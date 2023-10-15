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
move_delay = 0
rotate_delay = 0

running = True

while running:
    # Calculate delta time
    g_controller.update_delta_time()
    
    # Take player input
    key = pygame.key.get_pressed()
    move_delay -= 1 
    rotate_delay -= 1
    
    if (key[pygame.K_RIGHT] == True) and (move_delay < 0):
        p_controller.shift_piece_horizontally(1)
        move_delay = 75 * 32 / bu.GRID_SIZE
        
    if key[pygame.K_LEFT] == True and (move_delay < 0):
        p_controller.shift_piece_horizontally(-1)
        move_delay = 75 * 32 / bu.GRID_SIZE
        
    if key[pygame.K_x] == True and (rotate_delay < 0):
        p_controller.rotate_piece(1)
        rotate_delay = 120 * 32 / bu.GRID_SIZE
        
    if key[pygame.K_z] == True and (rotate_delay < 0):
        p_controller.rotate_piece(-1)
        rotate_delay = 120 * 32 / bu.GRID_SIZE
    
    # DEBUG EVENTS
    if key[pygame.K_a] == True:
        g_controller.set_drop_speed(20)
    
    if key[pygame.K_s] == True:
        g_controller.set_drop_speed(1)
    
    # Check pygame events
    for event in pygame.event.get():
        # Check if user has quit the window
        if event.type == pygame.QUIT:
            running = False
            
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                p_controller.hard_drop_piece()
                g_controller.new_piece_and_timer()
                
            if event.key == pygame.K_DOWN:
                p_controller.hard_drop_piece()
                
            if event.key == pygame.K_r:
                p_controller.restart_board()
                
            if event.key == pygame.K_LSHIFT:
                p_controller.hold_piece()
    
    ##### GAME LOGIC CALCULATIONS #####
    g_controller.run_timed_game_logic()
    
    # Clear window and draw window background       
    window.fill(0)
    
    # Draw board background
    window.blit(board_surface, (0, 0))
    board_surface.fill(0)
    board_surface.fill((0, 0, 0, bu.BACKGROUND_ALPHA), pygame.Rect(bu.BOARD_LEFT_BUFFER, bu.BOARD_TOP_BUFFER, bu.BOARD_WIDTH, bu.BOARD_HEIGHT))

    # Draw board grids 
    #window.blit(grid_surface, (0, 0))   
    #grid_surface.fill(0) 
    bu.draw_grids(board_surface)
    
    # Draw all pieces
    #window.blit(piece_surface, (0, 0))
    #piece_surface.fill(0)
    g_controller.draw_pieces(board_surface)
    
    # Draw fps counter
    board_surface.blit(g_controller.fps_string, (bu.SCR_WIDTH - (bu.GRID_SIZE * 3), bu.GRID_SIZE - bu.GRID_SIZE / 2))

    # Update window
    pygame.display.flip()
    
    # Increment frames that have passed
    g_controller.frames += 1

pygame.quit()