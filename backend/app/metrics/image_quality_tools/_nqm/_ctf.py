import numpy as np


def ctf(f_r):
    s = f_r.shape
    f_r = f_r.flatten()

    y = 1.0 / (200 * (2.6 * (0.0192 + 0.114 * f_r) * np.exp(-(0.114 * f_r)**1.1)))

    return y.reshape(y, s[0], s[1])
