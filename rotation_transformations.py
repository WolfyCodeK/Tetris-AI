def __is_side_square(x, y) -> bool:
        return (not x) ^ (not y)
    
def rotate_anticlockwise(shape, is_IPiece = False):
    if (not is_IPiece):
        for i in range(len(shape)):
            if (__is_side_square(shape[i][0], shape[i][1])):
                if (shape[i][1] == -1):
                    shape[i][0] = shape[i][0] - 1
                    shape[i][1] = shape[i][1] + 1
                elif (shape[i][1] == 1):
                    shape[i][0] = shape[i][0] + 1
                    shape[i][1] = shape[i][1] - 1  
                elif (shape[i][0] == -1):
                    shape[i][0] = shape[i][0] + 1
                    shape[i][1] = shape[i][1] + 1 
                elif (shape[i][0] == 1):
                    shape[i][0] = shape[i][0] - 1
                    shape[i][1] = shape[i][1] - 1
            else:
                if (shape[i][0] == -1) and ((shape[i][1] == -1)):
                    shape[i][1] = shape[i][1] + 2
                elif (shape[i][0] == 1) and ((shape[i][1] == -1)):
                    shape[i][0] = shape[i][0] - 2
                elif (shape[i][0] == 1) and ((shape[i][1] == 1)):
                    shape[i][1] = shape[i][1] - 2
                elif (shape[i][0] == -1) and ((shape[i][1] == 1)):
                    shape[i][0] = shape[i][0] + 2
    else:
        if (shape[0][0] == 0 and shape[0][1] == 0):
            # STATE 1
            for i in range(len(shape)):
                if (shape[i][0] == -1):
                    shape[i][0] = shape[i][0] + 1
                    shape[i][1] = shape[i][1] + 2
                elif (shape[i][0] == 0):
                    shape[i][1] = shape[i][1] + 1
                elif (shape[i][0] == 1):
                    shape[i][0] = shape[i][0] - 1
                elif (shape[i][0] == 2):
                    shape[i][0] = shape[i][0] - 2
                    shape[i][1] = shape[i][1] - 1
        elif (shape[0][0] == 1 and shape[0][1] == 0):
            # STATE 2
            for i in range(len(shape)):
                if (shape[i][1] == -1):
                    shape[i][0] = shape[i][0] - 2
                    shape[i][1] = shape[i][1] + 1
                elif (shape[i][1] == 0):
                    shape[i][0] = shape[i][0] - 1
                elif (shape[i][1] == 1):
                    shape[i][1] = shape[i][1] - 1
                elif (shape[i][1] == 2):
                    shape[i][0] = shape[i][0] + 1
                    shape[i][1] = shape[i][1] - 2
        elif (shape[0][0] == 1 and shape[0][1] == 1):
            # STATE 3
            for i in range(len(shape)):
                if (shape[i][0] == -1):
                    shape[i][0] = shape[i][0] + 2
                    shape[i][1] = shape[i][1] + 1
                elif (shape[i][0] == 0):
                    shape[i][0] = shape[i][0] + 1
                elif (shape[i][0] == 1):
                    shape[i][1] = shape[i][1] - 1
                elif (shape[i][0] == 2):
                    shape[i][0] = shape[i][0] - 1
                    shape[i][1] = shape[i][1] - 2
        elif (shape[0][0] == 0 and shape[0][1] == 1):
            # STATE 4
            for i in range(len(shape)):
                if (shape[i][1] == -1):
                    shape[i][0] = shape[i][0] - 1
                    shape[i][1] = shape[i][1] + 2
                elif (shape[i][1] == 0):
                    shape[i][1] = shape[i][1] + 1
                elif (shape[i][1] == 1):
                    shape[i][0] = shape[i][0] + 1
                elif (shape[i][1] == 2):
                    shape[i][0] = shape[i][0] + 2
                    shape[i][1] = shape[i][1] - 1
                    
    return shape

def rotate_clockwise(shape, is_IPiece = False):
    if (not is_IPiece):
        for i in range(len(shape)):
            if (__is_side_square(shape[i][0], shape[i][1])):
                if (shape[i][1] == -1):
                    shape[i][0] = shape[i][0] + 1
                    shape[i][1] = shape[i][1] + 1
                elif (shape[i][1] == 1):
                    shape[i][0] = shape[i][0] - 1
                    shape[i][1] = shape[i][1] - 1  
                elif (shape[i][0] == -1):
                    shape[i][0] = shape[i][0] + 1
                    shape[i][1] = shape[i][1] - 1 
                elif (shape[i][0] == 1):
                    shape[i][0] = shape[i][0] - 1
                    shape[i][1] = shape[i][1] + 1
            else:
                if (shape[i][0] == -1) and ((shape[i][1] == -1)):
                    shape[i][0] = shape[i][0] + 2
                elif (shape[i][0] == 1) and ((shape[i][1] == -1)):
                    shape[i][1] = shape[i][1] + 2
                elif (shape[i][0] == 1) and ((shape[i][1] == 1)):
                    shape[i][0] = shape[i][0] - 2
                elif (shape[i][0] == -1) and ((shape[i][1] == 1)):
                    shape[i][1] = shape[i][1] - 2
    else:
        if (shape[0][0] == 0 and shape[0][1] == 0):
            # STATE 1
            for i in range(len(shape)):
                if (shape[i][0] == -1):
                    shape[i][0] = shape[i][0] + 2
                    shape[i][1] = shape[i][1] - 1
                elif (shape[i][0] == 0):
                    shape[i][0] = shape[i][0] + 1
                elif (shape[i][0] == 1):
                    shape[i][1] = shape[i][1] + 1
                elif (shape[i][0] == 2):
                    shape[i][0] = shape[i][0] - 1
                    shape[i][1] = shape[i][1] + 2
        elif (shape[0][0] == 1 and shape[0][1] == 0):
            # STATE 2
            for i in range(len(shape)):
                if (shape[i][1] == -1):
                    shape[i][0] = shape[i][0] + 1
                    shape[i][1] = shape[i][1] + 2
                elif (shape[i][1] == 0):
                    shape[i][1] = shape[i][1] + 1
                elif (shape[i][1] == 1):
                    shape[i][0] = shape[i][0] - 1
                elif (shape[i][1] == 2):
                    shape[i][0] = shape[i][0] - 2
                    shape[i][1] = shape[i][1] - 1
        elif (shape[0][0] == 1 and shape[0][1] == 1):
            # STATE 3
            for i in range(len(shape)):
                if (shape[i][0] == -1):
                    shape[i][0] = shape[i][0] + 1
                    shape[i][1] = shape[i][1] - 2
                elif (shape[i][0] == 0):
                    shape[i][1] = shape[i][1] - 1
                elif (shape[i][0] == 1):
                    shape[i][0] = shape[i][0] - 1
                elif (shape[i][0] == 2):
                    shape[i][0] = shape[i][0] - 2
                    shape[i][1] = shape[i][1] + 1
        elif (shape[0][0] == 0 and shape[0][1] == 1):
            # STATE 4
            for i in range(len(shape)):
                if (shape[i][1] == -1):
                    shape[i][0] = shape[i][0] + 2
                    shape[i][1] = shape[i][1] + 1
                elif (shape[i][1] == 0):
                    shape[i][0] = shape[i][0] + 1
                elif (shape[i][1] == 1):
                    shape[i][1] = shape[i][1] - 1
                elif (shape[i][1] == 2):
                    shape[i][0] = shape[i][0] - 1
                    shape[i][1] = shape[i][1] - 2
                    
    return shape