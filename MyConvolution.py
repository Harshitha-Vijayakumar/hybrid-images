import numpy as np

def convolve(image, kernel):
    img_h, img_w, img_c = image.shape
    kernel_h, kernel_w = kernel.shape
    h, w = kernel_h // 2, kernel_w // 2

    conv_img = np.zeros(image.shape)
    padded_image = np.pad(image, ((h, h), (w, w), (0,0)), mode='constant')

    for k in range(img_c):
      for i in range(img_h):
        for j in range(img_w):
            region = padded_image[i:i+kernel_h, j:j+kernel_w, k]
            conv_img[i, j, k] = np.sum(region * kernel)

    return conv_img