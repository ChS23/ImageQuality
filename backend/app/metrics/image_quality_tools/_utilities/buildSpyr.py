import numpy as np

from .sp1Filters import sp1Filters
from .maxPyrHt import maxPyrHt
from .corrDn import corrDn
from .buildSpyrLevs import build_spyr_levs


def build_spyr(
        im, ht='auto', filtfile='sp1Filters', edges='reflect1'
):
    lo0filt,hi0filt,lofilt,bfilts,steermtx,harmonics = sp1Filters()

    max_ht = maxPyrHt(im.shape, lofilt.shape[0])

    if ht == 'auto':
        ht = max_ht

    hi0 = corrDn(im, hi0filt, edges)
    lo0 = corrDn(im, lo0filt, edges)

    pyr, pind = build_spyr_levs(lo0, ht, lofilt, bfilts, edges)

    pyr = np.concatenate((hi0.flatten(), pyr))
    pind = np.vstack((hi0.shape, pind))

    return pyr, pind, steermtx, harmonics
