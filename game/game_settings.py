class GameSettings():
    #########################################################
    #------------------ GAMEPLAY SETTINGS ------------------#
    #########################################################

    # The larger the drop speed, the quicker pieces will fall
    drop_speed = 1 

    # How many pieces to show in the queue
    num_of_queue_to_show = 5

    #########################################################
    #------------------- VISUAL SETTINGS -------------------#
    #########################################################

    # Screen size constants
    LOWEST_SCREEN_SIZE = 3

    # How large the game window and board are
    screen_size = 14

    # If the fps counter should be displayed in the top right corner of the window
    show_fps_counter = True
    
    #########################################################
    #------------------- ERROR MESSAGES -------------------#
    #########################################################
    SCREEN_SIZE_MINIMUM_ERROR = "GameSettings.set_screen_size() -> screen size must be at least 1"
    
    def set_screen_size(size):
        if (size > 0):
            GameSettings.screen_size = size
        else:
            raise ValueError(GameSettings.SCREEN_SIZE_MINIMUM_ERROR)
            
        