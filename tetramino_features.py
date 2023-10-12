import board_utils as bu

Z_PIECE_PID = 1
Z_PIECE_COLOUR = (255, 0, 0)
Z_PIECE_START_X = 4
Z_PIECE_START_Y = int(-(bu.pixel_to_grid_size(bu.DROP_HEIGHT)))

S_PIECE_PID = 2
S_PIECE_COLOUR = (0, 255, 0)
S_PIECE_START_X = 4
S_PIECE_START_Y = int(-(bu.pixel_to_grid_size(bu.DROP_HEIGHT)))

L_PIECE_PID = 3
L_PIECE_COLOUR = (255,120,0)
L_PIECE_START_X = 4
L_PIECE_START_Y = int(-(bu.pixel_to_grid_size(bu.DROP_HEIGHT)))

J_PIECE_PID = 4
J_PIECE_COLOUR = (0,0,255)
J_PIECE_START_X = 4
J_PIECE_START_Y = int(-(bu.pixel_to_grid_size(bu.DROP_HEIGHT)))

T_PIECE_PID = 5
T_PIECE_COLOUR = (255,20,147)
T_PIECE_START_X = 4
T_PIECE_START_Y = int(-(bu.pixel_to_grid_size(bu.DROP_HEIGHT)))