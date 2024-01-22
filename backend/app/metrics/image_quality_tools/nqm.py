import numpy as np

from ._nqm.nqm_modified import nqm as nqm_modified


def nqm(
        reference_image: np.ndarray,
        query_image: np.ndarray
):
    viewing_angle = 1/3.5 * 180 / np.pi

    dim = np.sqrt(np.prod(reference_image.shape))

    return nqm_modified(reference_image, query_image, viewing_angle, dim)
