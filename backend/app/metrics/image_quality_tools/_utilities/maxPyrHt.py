import numpy as np


def maxPyrHt(
        imsz,
        filtsz
):
    imsz = np.array(imsz).flatten()
    filtsz = np.array(filtsz).flatten()

    if np.any(imsz == 1):
        imsz = np.prod(imsz)
        filtsz = np.prod(filtsz)
    elif np.any(filtsz == 1):
        filtsz = np.array([filtsz[0], filtsz[0]])

    if np.any(imsz < filtsz):
        height = 0
    else:
        height = 1 + maxPyrHt(np.floor(imsz / 2), filtsz)

    return height
