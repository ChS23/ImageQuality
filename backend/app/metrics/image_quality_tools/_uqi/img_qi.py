import numpy as np

from .._functions import filter2


def uqi(
        img1: np.ndarray,
        img2: np.ndarray,
        block_size: int = 8,
) -> tuple:

    if img1.shape != img2.shape:
        raise ValueError('Images must have the same dimensions.')

    N = block_size ** 2
    sum2_filter = np.ones((block_size, block_size))

    img1_sq = img1 ** 2
    img2_sq = img2 ** 2
    img12 = img1 * img2

    img1_sum = filter2(sum2_filter, img1)
    img2_sum = filter2(sum2_filter, img2)
    img1_sq_sum = filter2(sum2_filter, img1_sq)
    img2_sq_sum = filter2(sum2_filter, img2_sq)
    img12_sum = filter2(sum2_filter, img12)

    img12_sum_mul = img1_sum * img2_sum
    img12_sq_sum_mul = img1_sum ** 2 + img2_sum ** 2
    numerator = 4 * (N * img12_sum - img12_sum_mul) * img12_sum_mul
    denominator1 = N * (img1_sq_sum + img2_sq_sum) - img12_sq_sum_mul
    denominator = denominator1 * img12_sq_sum_mul

    quality_map = np.ones(denominator.shape)
    index = np.logical_and(denominator1 == 0, img12_sq_sum_mul != 0)
    quality_map[index] = 2 * img12_sum_mul[index] / img12_sq_sum_mul[index]
    index = denominator != 0
    quality_map[index] = numerator[index] / denominator[index]

    quality = np.mean(quality_map)

    return quality, quality_map
