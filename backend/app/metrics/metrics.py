import numpy as np
import piq
import sewar
import torch
from PIL import Image
import torchvision.transforms as tt

from .image_quality_tools import mse, psnr, ssim, uqi, snr


metrics = {
    "PIQ": {
        "PSNR": piq.psnr,
        "SSIM": piq.ssim,
        "MS-SSIM": piq.multi_scale_ssim,
        "IW-SSIM": piq.information_weighted_ssim,
        "VIFp": piq.vif_p,
        "FSIM": piq.fsim,
        "SR-SIM": piq.srsim,
        "GMSD": piq.gmsd,
        "MS-GMSD": piq.multi_scale_gmsd,
        "VSI": piq.vsi,
        "DSS": piq.dss,
        "HaarPSI": piq.haarpsi,
        "MDSI": piq.mdsi,
        # "LPIPS": piq.LPIPS(),
        # "PieAPP": piq.PieAPP(),
        # "DISTS": piq.DISTS(),
    },
    "sewar": {
        "MSE": sewar.mse,
        "RMSE": sewar.rmse,
        "PSNR": sewar.psnr,
        "SSIM": sewar.ssim,
        "UQI": sewar.uqi,
        "MS-SSIM": sewar.msssim,
        "ERGAS": sewar.ergas,
        "SCC": sewar.scc,
        "RASE": sewar.rase,
        "SAM": sewar.sam,
        "D_lambda": sewar.d_lambda,
        # "D_S": sewar.d_s,
        # "QNR": sewar.qnr,
        "VIFp": sewar.vifp,
        "PSNR-B": sewar.psnrb
    },
    "image_quality_tools": {
        "MSE": mse,
        "PSNR": psnr,
        "SSIM": ssim,
        "UQI": uqi,
        "SNR": snr,
    },
}


def __preprocess(image1: Image, image2: Image, package: str) -> tuple:
    match package:
        case "PIQ":
            return tt.ToTensor()(image1).unsqueeze(0), tt.ToTensor()(image2).unsqueeze(0)
        case "sewar":
            return np.array(image1), np.array(image2)
        case "image_quality_tools":
            return np.array(image1.convert("L")), np.array(image2.convert("L"))


def calculate_metric(image1: Image, image2: Image, package: str, metric: str) -> float or None:
    if package not in metrics.keys():
        raise ValueError(f"Package {package} not found")

    if metric not in metrics[package].keys():
        raise ValueError(f"Metric {metric} not found")

    result = metrics[package][metric](*__preprocess(image1, image2, package))

    print(type(result))

    if isinstance(result, torch.Tensor):
        return float(result.item())
    if isinstance(result, np.float64):
        return float(result)
    if isinstance(result, tuple):
        return float(result[0])
    if isinstance(result, np.complex_):
        return float(result.real)

    return None
