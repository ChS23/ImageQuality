import numpy as np


def ind2wtree(pyr, ind):
    # This function converts the output of Eero Simoncelli's pyramid routines into subbands in a cell array
    C = pyr
    S = ind

    offset = 0
    numsubs = ind.shape[0]
    wtree = []

    for i in range(numsubs):
        wtree.append(np.reshape(C[offset:offset + np.prod(S[i])], S[i]))
        offset += np.prod(S[i])

    return wtree
