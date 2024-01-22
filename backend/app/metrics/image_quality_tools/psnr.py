import numpy as np

from .mse import mse


def psnr(
        img1: np.ndarray,
        img2: np.ndarray
) -> float:
    return 10 * np.log10(255 * 255 / (mse(img1, img2) + 1e-10))
