import numpy as np


def refparams_vecgsm(org, subands, M):
    ssarr = []
    l_arr = np.zeros((len(subands), M * M))
    cu_arr = []

    for i in range(len(subands)):
        sub = subands[i]
        y = org[sub]

        sizey = np.floor(np.array(y.shape) / M) * M  # crop to exact multiple size
        y = y[:int(sizey[0]), :int(sizey[1])]

        # Collect MxM blocks. Rearrange each block into an
        # M^2 dimensional vector and collect all such vectors.
        # Collect ALL possible MXM blocks (even those overlapping) from the subband
        temp = np.empty((0, M * M))
        for j in range(M):
            for k in range(M):
                temp = np.concatenate((temp, y[k:y.shape[0] - (M - k), j:y.shape[1] - (M - j)].reshape(1, -1)))

        # Estimate mean and covariance
        mcu = np.mean(temp, axis=1)
        cu = ((temp - mcu[:, None]) @ (temp - mcu[:, None]).T) / temp.shape[1]  # covariance matrix for U

        # Collect MxM blocks as above. Use ONLY non-overlapping blocks to
        # calculate the S field
        temp = np.empty((0, M * M))
        for j in range(M):
            for k in range(M):
                temp = np.concatenate((temp, y[k::M, j::M].reshape(1, -1)))

        # Calculate the S field
        ss = np.linalg.solve(cu, temp.T)
        ss = np.sum(ss * temp, axis=0) / (M * M)
        ss = ss.reshape(sizey // M)

        # Eigen-decomposition
        _, d, v = np.linalg.svd(cu)
        l_arr[i, :] = d

        # Rearrange for output
        ssarr.append(ss)
        cu_arr.append(cu)

    return ssarr, l_arr, cu_arr
