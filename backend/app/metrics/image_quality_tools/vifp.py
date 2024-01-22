import numpy as np
from ._functions import gauss2D, filter2


def vifp(
        ref: np.ndarray,
        dist: np.ndarray,
):
    sigma_nsq = 2
    num = 0
    den = 0
    for scale in range(1, 5):
        N = 2 ** (4 - scale + 1) + 1
        win = gauss2D(shape=(N, N), sigma=N / 5)

        if scale > 1:
            ref = filter2(ref, win)
            dist = filter2(dist, win)
            ref = ref[1:2:-1, 1:2:-1]
            dist = dist[1:2:-1, 1:2:-1]

        mu1 = filter2(win, ref)
        mu2 = filter2(win, dist)
        mu1_sq = mu1 * mu1
        mu2_sq = mu2 * mu2
        mu1_mu2 = mu1 * mu2
        sigma1_sq = filter2(win, ref * ref) - mu1_sq
        sigma2_sq = filter2(win, dist * dist) - mu2_sq
        sigma12 = filter2(win, ref * dist) - mu1_mu2

        sigma1_sq[sigma1_sq < 0] = 0
        sigma2_sq[sigma2_sq < 0] = 0

        g = sigma12 / (sigma1_sq + 1e-10)
        sv_sq = sigma2_sq - g * sigma12

        g[sigma1_sq < 1e-10] = 0
        sv_sq[sigma1_sq < 1e-10] = sigma2_sq[sigma1_sq < 1e-10]
        sigma1_sq[sigma1_sq < 1e-10] = 0

        g[sigma2_sq < 1e-10] = 0
        sv_sq[sigma2_sq < 1e-10] = 0

        sv_sq[g < 0] = sigma2_sq[g < 0]
        g[g < 0] = 0
        sv_sq[sv_sq <= 1e-10] = 1e-10

        num += np.sum(np.log10(1 + g * g * sigma1_sq / (sv_sq + sigma_nsq)))
        den += np.sum(np.log10(1 + sigma1_sq / sigma_nsq))

    return num / den
