import numpy as np

from ._ifc.ifcvec import ifc as _ifc


def ifc(
        reference_image: np.ndarray,
        query_image: np.ndarray,
):
    return _ifc(reference_image, query_image)
