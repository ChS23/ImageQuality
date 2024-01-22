import numpy as np

from .._utilities.buildSpyr import build_spyr
from .._vif.ind2wtree import ind2wtree
from .refparams_vecgsm import refparams_vecgsm
from .distsub_est_m import vifsub_est_M


def ifc(
        imorg: np.ndarray,
        imdist: np.ndarray
):
    M = 3
    subbands = [4, 7, 10, 13, 16, 19, 22, 25]

    # Do wavelet decomposition. This requires the Steerable Pyramid. You can
    # use your own wavelet as long as the cell arrays org and dist contain
    # corresponding subbands from the reference and the distorted images
    # respectively.
    pyr, pind = build_spyr(imorg, 4, 'sp5Filters', 'reflect1')
    org = ind2wtree(pyr, pind)
    pyr, pind = build_spyr(imdist, 4, 'sp5Filters', 'reflect1')
    dist = ind2wtree(pyr, pind)

    g_all, vv_all = vifsub_est_M(org, dist, subbands, M)
    ssarr, larr, cuarr = refparams_vecgsm(org, subbands, M)

    vvtemp = [None] * (max(subbands) + 1)
    ggtemp = [None] * (max(subbands) + 1)
    for kk in range(len(subbands)):
        vvtemp[subbands[kk]] = vv_all[kk]
        ggtemp[subbands[kk]] = g_all[kk]

    num = np.zeros(len(subbands))
    for i in range(len(subbands)):
        sub = subbands[i]
        g = ggtemp[sub]
        vv = vvtemp[sub]
        ss = ssarr[sub]
        lambda_val = larr[sub]
        cu = cuarr[sub]

        # How many eigenvalues to sum over. Default is 1.
        neigvals = 1

        # Compute the size of the window used in the distortion channel estimation
        lev = np.ceil((sub - 1) / 6)
        winsize = 2**lev + 1
        offset = (winsize - 1) / 2
        offset = np.ceil(offset / M)

        # Select only the valid portion of the output
        g = g[offset+1:, offset+1:]
        vv = vv[offset+1:, offset+1:]
        ss = ss[offset+1:, offset+1:]

        # IFC
        temp1 = 0
        for j in range(len(lambda_val)):
            temp1 += np.sum(np.sum(np.log2(1 + g**2 * ss * lambda_val[j] / (vv + 1e-10))))

        num[i] = temp1

    # Compute IFC and normalize to the size of the image
    ifc_value = np.sum(num) / np.prod(np.array(imorg.shape))
    return ifc_value
