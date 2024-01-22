import numpy as np


def vsnr_modified(
        src_img: np.ndarray,
        dst_img: np.ndarray,
        alpha: float = 0.04,
        viewing_params=None
) -> float:
    if src_img.shape != dst_img.shape:
        raise ValueError('Images must have the same shape.')

    vsnr_data = {
        'b': 0,
        'k': 0.02874,
        'g': 2.2,
        'r': 96,
        'v': 19.1,
        'num_levels': 5
    }

    vsnr_data['filter_gains '] = 2 ** np.arange(0, vsnr_data['num_levels'])

    vsnr_data['fs'] = (vsnr_data['r'] * vsnr_data['v'] * np.tan(np.pi / 180)
                       * (2 ** -(np.arange(0, vsnr_data['num_levels']) + 1)))
