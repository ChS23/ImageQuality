import numpy as np


def wsnr(
        orig: np.ndarray,
        dith: np.ndarray,
        nfreq=60
):
    if orig.dtype != dith.dtype:
        raise ValueError("Input images must have the same dtype.")

    if orig.shape != dith.shape:
        raise ValueError("Input images must have the same dimensions.")

    if orig.ndim != 2 or dith.ndim != 2:
        raise ValueError("Input images must be 2D.")

    x, y = orig.shape

    # generate mesh
    xplane, yplane = np.meshgrid(np.arange(-y / 2 + 0.5, y / 2 - 0.5), np.arange(-x / 2 + 0.5, x / 2 - 0.5))

    plane = (xplane + 1j * yplane) / x * 2 * nfreq
    radfreq = np.abs(plane)

    w = 0.7
    s = (1 - w) / 2 * np.cos(4 * np.angle(plane)) + (1 + w) / 2
    radfreq /= s

    # Now generate the CSF
    csf = 2.6 * (0.0192 + 0.114 * radfreq) * np.exp(-(0.114 * radfreq) ** 1.1)
    f = np.where(radfreq < 7.8909)
    csf[f] = 0.9809 + np.zeros(np.shape(f))

    # Find weighted SNR in frequency domain.  Note that, because we are not
    # weighting the signal, we compute signal power in the spatial domain.
    # This requires us to multiply by the image size in pixels to get the
    # signal power in the freqency domain for division.
    err = orig - dith
    err_wt = np.fft.fftshift(np.fft.fft2(err)) * csf
    im = np.fft.fft2(orig)

    mse = np.sum(np.sum(err_wt * np.conj(err_wt)))  # weighted error power
    mss = np.sum(np.sum(im * np.conj(im)))  # signal power
    return 10 * np.log10(mss / mse)  # compute SNR
