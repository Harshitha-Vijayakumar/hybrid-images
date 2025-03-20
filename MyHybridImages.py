import numpy as np
from MyConvolution import convolve

def myHybridImages(low_img, low_sigma, high_img, high_sigma):
    low_kernel = makeGaussianKernel(low_sigma)
    low_image = convolve(low_img, low_kernel)

    high_kernel = makeGaussianKernel(high_sigma)
    high_image = high_img - convolve(high_img, high_kernel)

    myHybridImage = low_image + high_image

    return myHybridImage


def makeGaussianKernel(sigma):
    size = int(np.floor(8 * sigma + 1))
    if size % 2 == 0:
        size += 1
    
    kernel = np.zeros((size, size))
    center = size // 2
    x = np.arange(-center, center + 1)
    y = np.arange(-center, center + 1)

    for i in range(size):
        for j in range(size):
            kernel[i, j] = (1 / (2 * np.pi * sigma ** 2)) * np.exp(-(x[i]**2 + y[j]**2) / (2 * sigma**2))

    return kernel/np.sum(kernel)

