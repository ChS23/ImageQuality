import numpy as np

from ._functions import gauss2D, filter2


def ssim_modified(
        img1: np.ndarray,
        img2: np.ndarray,
        K=None,
        window=None,
        L=None
) -> tuple:

    if img1.shape != img2.shape:
        raise ValueError('Input images must have the same dimensions.')

    if img1.dtype != img2.dtype:
        raise ValueError('Input images must have the same dtype.')

    if img1.ndim != 2 or img2.ndim != 2:
        raise ValueError('Input images must be 2D.')

    M, N = img1.shape

    if K is None:
        K = [0.01, 0.03]

    if window is None:
        window = gauss2D((11, 11), 1.5)
        window /= np.sum(window)

    if L is None:
        L = 255

    C1 = (K[0] * L) ** 2
    C2 = (K[1] * L) ** 2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    mu1 = filter2(img1, window)
    mu2 = filter2(img2, window)

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = filter2(window, img1 ** 2) - mu1_sq
    sigma2_sq = filter2(window, img2 ** 2) - mu2_sq
    sigma1 = np.sqrt(np.abs(sigma1_sq))
    sigma2 = np.sqrt(np.abs(sigma2_sq))
    sigma12 = filter2(window, img1 * img2) - mu1_mu2

    if C1 > 0 and C2 > 0:
        M = (2 * mu1_mu2 + C1) / (mu1_sq + mu2_sq + C1)
        V = (2 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2)
        R = (sigma12 + C2 / 2) / (sigma1 * sigma2 + C2 / 2)
        ssim_map = M * V * R
    else:
        ssim_ln = 2 * mu1_mu2
        ssim_ld = mu1_sq + mu2_sq
        M = np.ones(mu1.shape)
        index_l = ssim_ld > 0
        M[index_l] = ssim_ln[index_l] / ssim_ld[index_l]

        ssim_cn = 2 * sigma1 * sigma2
        ssim_cd = sigma1_sq + sigma2_sq
        V = np.ones(mu1.shape)
        index_c = ssim_cd > 0
        V[index_c] = ssim_cn[index_c] / ssim_cd[index_c]

        ssim_sn = sigma12
        ssim_sd = sigma1 * sigma2
        R = np.ones(mu1.shape)
        index1 = sigma1 > 0
        index2 = sigma2 > 0
        index_s = (index1 * index2) > 0
        R[index_s] = ssim_sn[index_s] / ssim_sd[index_s]
        index_s = (index1 * (~index2)) > 0
        R[index_s] = 0
        index_s = ((~index1) * index2) > 0
        R[index_s] = 0

        ssim_map = M * V * R

    ssim = np.mean(ssim_map)

    return ssim, ssim_map, [M, V, R], [np.mean(M), np.mean(V), np.mean(R)]
