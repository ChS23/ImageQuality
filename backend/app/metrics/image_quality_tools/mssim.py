import numpy as np
from scipy.signal import convolve2d

from ._functions import gauss2D
from .ssim_modified import ssim_modified


def mssim(
        img1: np.ndarray,
        img2: np.ndarray,
        nlevs: int = 5,
        K=None,
        lpf=None
) -> tuple:

    if img1.shape != img2.shape:
        raise ValueError('Input images must have the same dimensions.')

    if img1.dtype != img2.dtype:
        raise ValueError('Input images must have the same dtype.')

    if K is None:
        K = [0.01, 0.03]

    if lpf is None:
        lpf = np.array(
            [
                [0.037828455507260, -0.023849465019560, -0.110624404418440],
                [0.377402855612830, 0.852698679008890, 0.377402855612830],
                [-0.110624404418440, -0.023849465019560, 0.037828455507260]
            ]
        )

        lpf /= np.sum(lpf)

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    window = gauss2D((11, 11), 1.5)
    window /= np.sum(window)

    ssim_v = np.zeros((nlevs,))
    ssim_r = np.zeros((nlevs,))

    _, _, _, comp_ssim = ssim_modified(img1, img2, K)
    ssim_v[0] = comp_ssim[1]
    ssim_r[0] = comp_ssim[2]

    ssim_m = comp_ssim[0]
    for s in range(0, nlevs):
        img1 = convolve2d(img1, lpf, mode='same', boundary='symm')
        img2 = convolve2d(img2, lpf, mode='same', boundary='symm')

        img1 = img1[::2, ::2]
        img2 = img2[::2, ::2]

        _, _, _, comp_ssim = ssim_modified(img1, img2, K)
        ssim_m = comp_ssim[0]
        ssim_v[s+1] = comp_ssim[1]
        ssim_r[s+1] = comp_ssim[2]

    alpha = 0.1333
    beta = [0.0448, 0.2856, 0.3001, 0.2363, 0.1333]
    gamma = [0.0448, 0.2856, 0.3001, 0.2363, 0.1333]

    comp = np.array([ssim_m ** alpha, np.prod(ssim_v ** beta), np.prod(ssim_r ** gamma)])

    detail = np.concatenate(([ssim_m], ssim_v, ssim_r))

    mssim_value = np.prod(comp)

    return mssim_value, comp, detail
