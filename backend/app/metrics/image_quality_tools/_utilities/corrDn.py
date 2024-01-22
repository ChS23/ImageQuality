import numpy as np


def corrDn(im, filt, edges=None, step=None, start=None, stop=None):
    if edges is not None and edges != 'reflect1':
        raise ValueError('Invalid edges argument')

    if step is None:
        step = [1, 1]

    if start is None:
        start = [1, 1]

    if stop is None:
        stop = [im.shape[0], im.shape[1]]

    filt = np.flipud(np.fliplr(filt))

    tmp = np.correlate(im, filt, 'full')
    return tmp[start[0]:stop[0]:step[0], start[1]:stop[1]:step[1]]
