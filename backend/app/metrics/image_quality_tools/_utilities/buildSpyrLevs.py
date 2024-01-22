import numpy as np

from .corrDn import corrDn


def build_spyr_levs(lo0,ht,lofilt,bfilts,edges):

    if ht <= 0:
        pyr = lo0.flatten()
        pind = lo0.shape
    else:
        bfiltsz = int(round(np.sqrt(bfilts.shape[0])))

        bands = np.zeros((np.prod(lo0.shape), bfilts.shape[1]))
        bind = np.zeros((bfilts.shape[1], 2), dtype=int)

        for b in range(bfilts.shape[1]):
            filt = np.reshape(bfilts[:, b], (bfiltsz, bfiltsz))
            band = corrDn(lo0, filt, edges)
            bands[:, b] = band.flatten()
            bind[b, :] = band.shape

        lo = corrDn(lo0, lofilt, edges, [2, 2], [1, 1])

        npyr, nind = build_spyr_levs(lo, ht-1, lofilt, bfilts, edges)

        pyr = np.concatenate((bands.flatten(), npyr))
        pind = np.vstack((bind, nind))

    return pyr, pind
