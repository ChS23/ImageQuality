import numpy as np

from .._utilities.corrDn import corrDn


def vifsub_est_M(org, dist, subbands, M):
    # Uses convolution for determining the parameters of the distortion channel

    tol = 1e-15  # Tolerance for zero variance. Variance below this is set to zero, and zero is set to this value to avoid numerical issues.

    g_all = []
    vv_all = []

    for i in range(len(subbands)):
        sub = subbands[i]
        y = org[sub].copy()
        yn = dist[sub].copy()

        # Compute the size of the window used in the distortion channel estimation
        lev = np.ceil((sub - 1) / 6)
        winsize = 2**lev + 1
        offset = (winsize - 1) // 2
        win = np.ones((winsize, winsize))

        # Force subband size to be a multiple of M
        newsize = np.floor(np.array(y.shape) / M) * M
        y = y[:int(newsize[0]), :int(newsize[1])]
        yn = yn[:int(newsize[0]), :int(newsize[1])]

        # Correlation with downsampling. This is faster than downsampling after
        # computing full correlation.
        winstep = [M, M]
        winstart = [1, 1] * (np.floor(np.array(M) / 2).astype(int)) + 1
        winstop = np.array(y.shape) - np.ceil(np.array(M) / 2).astype(int) + 1

        # Mean
        mean_x = corrDn(y, win / np.sum(win), 'reflect1', winstep, winstart, winstop)
        mean_y = corrDn(yn, win / np.sum(win), 'reflect1', winstep, winstart, winstop)

        # Covariance
        cov_xy = corrDn(y * yn, win, 'reflect1', winstep, winstart, winstop) - np.sum(win) * mean_x * mean_y

        # Variance
        ss_x = corrDn(y**2, win, 'reflect1', winstep, winstart, winstop) - np.sum(win) * mean_x**2
        ss_y = corrDn(yn**2, win, 'reflect1', winstep, winstart, winstop) - np.sum(win) * mean_y**2

        # Get rid of numerical problems, very small negative numbers, or very
        # small positive numbers, or other theoretical impossibilities.
        ss_x[ss_x < 0] = 0
        ss_y[ss_y < 0] = 0

        # Regression
        g = cov_xy / (ss_x + tol)

        # Variance of error in regression
        vv = (ss_y - g * cov_xy) / (np.sum(win))

        # Get rid of numerical problems, very small negative numbers, or very
        # small positive numbers, or other theoretical impossibilities.
        g[ss_x < tol] = 0
        vv[ss_x < tol] = ss_y[ss_x < tol]
        ss_x[ss_x < tol] = 0

        g[ss_y < tol] = 0
        vv[ss_y < tol] = 0

        # Constrain g to be non-negative.
        vv[g < 0] = ss_y[g < 0]
        g[g < 0] = 0

        # Take care of numerical errors, vv could be very small negative
        vv[vv <= tol] = tol

        g_all.append(g)
        vv_all.append(vv)

    return g_all, vv_all
