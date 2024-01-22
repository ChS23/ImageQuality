import numpy as np


def mse(
        img1: np.ndarray,
        img2: np.ndarray,
):
    return np.mean((img1.flatten() - img2.flatten()) ** 2)
