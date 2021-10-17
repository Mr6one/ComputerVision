import numpy as np
from numba import jit

import warnings
from tqdm.notebook import tqdm
warnings.filterwarnings('ignore')


idx_to_directions = {0: 'N', 1: 'E', 2: 'S', 3: 'W', 4: 'NE', 5: 'SE', 6: 'NW', 7: 'SW'}


@jit
def compute_gradient(cell, move_idx, diff_idx, weights):
    gradient = 0
    for weights_, move_id in zip(weights, move_idx):
        for weight, diff_id in zip(weights_, diff_idx):
            i1, j1 = move_id[0] + diff_id[0][0], move_id[1] + diff_id[0][1]
            i2, j2 = move_id[0] + diff_id[1][0], move_id[1] + diff_id[1][1]
            gradient += abs(cell[i1, j1] - cell[i2, j2]) / weight

    return gradient


@jit
def compute_green_gradients(cell):
    move_idx, diff_idx = [[2, 2], [2, 1], [2, 3]], [[[-1, 0], [1, 0]], [[-2, 0], [0, 0]]]
    weights = [[1, 1], [2, 2], [2, 2]]
    N = compute_gradient(cell, move_idx, diff_idx, weights)

    move_idx, diff_idx = [[2, 2], [1, 2], [3, 2]], [[[0, -1], [0, 1]], [[0, 2], [0, 0]]]
    weights = [[1, 1], [2, 2], [2, 2]]
    E = compute_gradient(cell, move_idx, diff_idx, weights)

    move_idx, diff_idx = [[2, 2], [2, 1], [2, 3]], [[[-1, 0], [1, 0]], [[2, 0], [0, 0]]]
    weights = [[1, 1], [2, 2], [2, 2]]
    S = compute_gradient(cell, move_idx, diff_idx, weights)

    move_idx, diff_idx = [[2, 2], [1, 2], [3, 2]], [[[0, -1], [0, 1]], [[0, -2], [0, 0]]]
    weights = [[1, 1], [2, 2], [2, 2]]
    W = compute_gradient(cell, move_idx, diff_idx, weights)

    move_idx, diff_idx = [[2, 2], [1, 3], [1, 2], [2, 3]], [[[-1, 1], [1, -1]]]
    weights = [[1] for i in range(4)]
    NE = compute_gradient(cell, move_idx, diff_idx, weights)

    move_idx, diff_idx = [[2, 2], [3, 3], [3, 2], [2, 3]], [[[1, 1], [-1, -1]]]
    weights = [[1] for i in range(4)]
    SE = compute_gradient(cell, move_idx, diff_idx, weights)

    move_idx, diff_idx = [[2, 2], [1, 1], [1, 2], [2, 1]], [[[1, 1], [-1, -1]]]
    weights = [[1] for i in range(4)]
    NW = compute_gradient(cell, move_idx, diff_idx, weights)

    move_idx, diff_idx = [[2, 2], [3, 1], [2, 1], [3, 2]], [[[-1, 1], [1, -1]]]
    weights = [[1] for i in range(4)]
    SW = compute_gradient(cell, move_idx, diff_idx, weights)

    return np.array([N, E, S, W, NE, SE, NW, SW])


@jit
def compute_reb_blue_gradients(cell):
    move_idx, diff_idx = [[2, 2], [2, 1], [2, 3]], [[[-1, 0], [1, 0]], [[-2, 0], [0, 0]]]
    weights = [[1, 1], [2, 2], [2, 2]]
    N = compute_gradient(cell, move_idx, diff_idx, weights)

    move_idx, diff_idx = [[2, 2], [1, 2], [3, 2]], [[[0, -1], [0, 1]], [[0, 2], [0, 0]]]
    weights = [[1, 1], [2, 2], [2, 2]]
    E = compute_gradient(cell, move_idx, diff_idx, weights)

    move_idx, diff_idx = [[2, 2], [2, 1], [2, 3]], [[[-1, 0], [1, 0]], [[2, 0], [0, 0]]]
    weights = [[1, 1], [2, 2], [2, 2]]
    S = compute_gradient(cell, move_idx, diff_idx, weights)

    move_idx, diff_idx = [[2, 2], [1, 2], [3, 2]], [[[0, -1], [0, 1]], [[0, -2], [0, 0]]]
    weights = [[1, 1], [2, 2], [2, 2]]
    W = compute_gradient(cell, move_idx, diff_idx, weights)

    move_idx, diff_idx = [[2, 2], [1, 3]], [[[-1, 1], [1, -1]]]
    weights = [[1, 1]]
    NE = compute_gradient(cell, move_idx, diff_idx, weights)

    move_idx, diff_idx = [[1, 2], [2, 3]], [[[0, 0], [1, -1]], [[0, 0], [-1, 1]]]
    weights = [[2] for i in range(4)]
    NE += compute_gradient(cell, move_idx, diff_idx, weights)

    move_idx, diff_idx = [[2, 2], [3, 3]], [[[1, 1], [-1, -1]]]
    weights = [[1, 1]]
    SE = compute_gradient(cell, move_idx, diff_idx, weights)

    move_idx, diff_idx = [[3, 2], [2, 3]], [[[0, 0], [-1, -1]], [[0, 0], [1, 1]]]
    weights = [[2] for i in range(4)]
    SE += compute_gradient(cell, move_idx, diff_idx, weights)

    move_idx, diff_idx = [[2, 2], [1, 1]], [[[1, 1], [-1, -1]]]
    weights = [[1, 1]]
    NW = compute_gradient(cell, move_idx, diff_idx, weights)

    move_idx, diff_idx = [[1, 2], [2, 1]], [[[0, 0], [1, 1]], [[0, 0], [-1, -1]]]
    weights = [[2] for i in range(4)]
    NW += compute_gradient(cell, move_idx, diff_idx, weights)

    move_idx, diff_idx = [[2, 2], [3, 1]], [[[-1, 1], [1, -1]]]
    weights = [[1, 1]]
    SW = compute_gradient(cell, move_idx, diff_idx, weights)

    move_idx, diff_idx = [[2, 1], [3, 2]], [[[0, 0], [1, -1]], [[0, 0], [-1, 1]]]
    weights = [[2] for i in range(4)]
    SW += compute_gradient(cell, move_idx, diff_idx, weights)

    return np.array([N, E, S, W, NE, SE, NW, SW])


@jit
def compute_optimal_subset(cell, compute_gradients):
    gradients = compute_gradients(cell)
    min_grad, max_grad = gradients.min(), gradients.max()
    thershold = 1.5 * min_grad + 0.5 * (min_grad + max_grad)
    idx = np.where(gradients <= thershold)[0]
    return [idx_to_directions[i] for i in idx]


@jit
def recovery(cell):
    optimal_subset = compute_optimal_subset(cell, compute_reb_blue_gradients)

    average_color_components = []
    for direction in optimal_subset:
        if direction == 'N':
            r, g, b = (cell[0, 2] + cell[2, 2]) / 2, cell[1, 2], (cell[1, 1] + cell[1, 3]) / 2
            average_color_components.append([r, g, b])
        elif direction == 'E':
            r, g, b = (cell[2, 2] + cell[2, 4]) / 2, cell[2, 3], (cell[1, 3] + cell[3, 3]) / 2
            average_color_components.append([r, g, b])
        elif direction == 'S':
            r, g, b = (cell[4, 2] + cell[2, 2]) / 2, cell[3, 2], (cell[3, 3] + cell[3, 1]) / 2
            average_color_components.append([r, g, b])
        elif direction == 'W':
            r, g, b = (cell[2, 2] + cell[2, 0]) / 2, cell[2, 1], (cell[3, 1] + cell[1, 1]) / 2
            average_color_components.append([r, g, b])
        elif direction == 'NE':
            r, g, b = (cell[0, 4] + cell[2, 2]) / 2, (cell[0, 3] + cell[1, 2] + cell[2, 3] + cell[1, 4]) / 4, cell[1, 3]
            average_color_components.append([r, g, b])
        elif direction == 'SE':
            r, g, b = (cell[4, 4] + cell[2, 2]) / 2, (cell[2, 3] + cell[3, 2] + cell[4, 3] + cell[3, 4]) / 4, cell[3, 3]
            average_color_components.append([r, g, b])
        elif direction == 'NW':
            r, g, b = (cell[0, 0] + cell[2, 2]) / 2, (cell[0, 1] + cell[1, 0] + cell[2, 1] + cell[1, 2]) / 4, cell[1, 1]
            average_color_components.append([r, g, b])
        elif direction == 'SW':
            r, g, b = (cell[4, 0] + cell[2, 2]) / 2, (cell[2, 1] + cell[3, 0] + cell[3, 2] + cell[4, 1]) / 4, cell[3, 1]
            average_color_components.append([r, g, b])

    if len(average_color_components) == 0:
        average_color_components.append([cell[2, 2]] * 3)
        optimal_subset = [0]

    r, g, b = np.array(average_color_components).sum(axis=0)
    '''
        код функции написан для обработки клетки с красным цветом в центре, но если в центре расположен синий, то
        красный и синий цвета меняются местами, поэтому если на вход поступила клетка с синим центром, то 
        в r содержится синий, а в b - красный, поэтому в color_2 будет содержатся синий
    '''
    color_1, color_2 = cell[2, 2] + (g - r) / len(optimal_subset), cell[2, 2] + (b - r) / len(optimal_subset)
    return color_1, color_2


@jit
def red_blue_recovery(cell):
    optimal_subset = compute_optimal_subset(cell, compute_green_gradients)

    average_color_components = []
    for direction in optimal_subset:
        if direction == 'N':
            r, g, b = cell[1, 2], (cell[0, 2] + cell[2, 2]) / 2, (cell[0, 1] + cell[0, 3] + cell[2, 1] + cell[2, 3]) / 4
            average_color_components.append([r, g, b])
        elif direction == 'E':
            r, g, b = (cell[1, 2] + cell[3, 2] + cell[1, 4] + cell[3, 4]) / 4, (cell[2, 2] + cell[2, 4]) / 2, cell[2, 3]
            average_color_components.append([r, g, b])
        elif direction == 'S':
            r, g, b = cell[3, 2], (cell[4, 2] + cell[2, 2]) / 2, (cell[4, 1] + cell[4, 3] + cell[2, 1] + cell[2, 3]) / 4
            average_color_components.append([r, g, b])
        elif direction == 'W':
            r, g, b = (cell[1, 2] + cell[3, 2] + cell[1, 0] + cell[3, 0]) / 4, (cell[2, 2] + cell[2, 0]) / 2, cell[2, 1]
            average_color_components.append([r, g, b])
        elif direction == 'NE':
            r, g, b = (cell[1, 2] + cell[1, 4]) / 2, cell[1, 3], (cell[0, 3] + cell[2, 3]) / 2
            average_color_components.append([r, g, b])
        elif direction == 'SE':
            r, g, b = (cell[3, 2] + cell[3, 4]) / 2, cell[3, 3], (cell[2, 3] + cell[4, 3]) / 2
            average_color_components.append([r, g, b])
        elif direction == 'NW':
            r, g, b = (cell[1, 0] + cell[1, 2]) / 2, cell[1, 1], (cell[0, 1] + cell[2, 1]) / 2
            average_color_components.append([r, g, b])
        elif direction == 'SW':
            r, g, b = (cell[3, 0] + cell[3, 2]) / 2, cell[3, 1], (cell[2, 1] + cell[4, 1]) / 2
            average_color_components.append([r, g, b])

    if len(average_color_components) == 0:
        average_color_components.append([cell[2, 2]] * 3)
        optimal_subset = [0]

    r, g, b = np.array(average_color_components).sum(axis=0)
    r, b = cell[2, 2] + (r - g) / len(optimal_subset), cell[2, 2] + (b - g) / len(optimal_subset)
    return r, b


# может быть неочевидно как одна и таже функция может давать разные результаты, см. комметарий к recovery
@jit
def green_blue_recovery(cell):
    g, b = recovery(cell)
    return g, b


@jit
def red_green_recovery(cell):
    g, r = recovery(cell)
    return r, g


@jit
def interpolate(cell, color, upper_color):
    if color == 'r':
        g, b = green_blue_recovery(cell)
        r = cell[2, 2]
    elif color == 'b':
        r, g = red_green_recovery(cell)
        b = cell[2, 2]
    else:
        if upper_color == 'b':
            cell = np.rot90(cell)
        r, b = red_blue_recovery(cell)
        g = cell[2, 2]

    return [r, g, b]


@jit
def reconstruct_image(transformed_image, bayer_filter, shape):
    reconstructed_image = np.zeros(shape)
    for i in tqdm(range(2, transformed_image.shape[0] - 2)):
        for j in range(2, transformed_image.shape[1] - 2):
            upper_color = bayer_filter[1][0]
            cell = transformed_image[i - 2:i + 3, j - 2:j + 3]
            reconstructed_image[i - 2, j - 2] = interpolate(cell, bayer_filter[0][0], upper_color)
            bayer_filter[:, (0, 1)] = bayer_filter[:, (1, 0)]

        bayer_filter[(0, 1), :] = bayer_filter[(1, 0), :]

    return reconstructed_image