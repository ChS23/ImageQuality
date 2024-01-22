import numpy as np

from .mse import mse


def snr(
        reference_image: np.ndarray,
        query_image: np.ndarray
) -> float:
    signal_value = np.mean((reference_image.flatten())**2)
    mse_value = mse(reference_image, query_image)

    return 10 * np.log10(signal_value / (mse_value + 1e-10))
