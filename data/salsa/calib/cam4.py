import numpy as np

extrinsics = np.array([
    [-0.638223, 0.767257, 0.0631593, 5.66734],
    [0.510238, 0.360137, 0.780999, 0.244033],
    [0.57648, 0.530677, -0.621331, 2.85218],
    [0, 0, 0, 1]
])

intrinsics = np.array([
    [0.995272 * 674.319, 0, 461.743],
    [0, 1 * 674.319, 398.09],
    [0, 0, 1]
])

distortion = np.array([-0.316926, 0.115751, 0, 0])