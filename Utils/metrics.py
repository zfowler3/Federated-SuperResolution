from skimage.metrics import structural_similarity as ssim
import numpy as np
from math import log10, sqrt

def mse(imageA, imageB):
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])
    return err

def PSNR(imageA, imageB):
    mse = np.mean((imageA - imageB) ** 2)
    if (mse == 0):
        return 100
    max_pixel = 255.0
    psnr = 20 * log10(max_pixel / sqrt(mse))
    return psnr

def ssim_(imageA, imageB):
    return ssim(imageA, imageB)