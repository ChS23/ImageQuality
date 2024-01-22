import numpy as np

from ._uqi.img_qi import uqi as _uqi


def uqi(
    reference_image: np.ndarray,
    distorted_image: np.ndarray
) -> tuple:
    return _uqi(reference_image, distorted_image)
