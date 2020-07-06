import cv2
import matplotlib.pyplot as plt
import numpy as np
import matplotlib

plt.ion()

matplotlib.rcParams["figure.figsize"] = (10.0, 10.0)
matplotlib.rcParams["image.cmap"] = "gray"

kernelSize = 3


def laplacianAndNormalize(image):
    # take grayscale of image first
    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # blur image
    img1 = cv2.GaussianBlur(img, (3, 3), 0, 0)
    # take laplacian for gradients
    laplacian = cv2.Laplacian(img1, cv2.CV_32F, ksize=kernelSize, scale=1, delta=0)
    # normalize image
    cv2.normalize(
        laplacian,
        dst=laplacian,
        alpha=0,
        beta=1,
        norm_type=cv2.NORM_MINMAX,
        dtype=cv2.CV_32F,
    )
    return laplacian


def pencilSketch(image, arguments=0):
    laplacian = laplacianAndNormalize(image)
    # picked these numbers from trial and error
    none_edge_pixels = laplacian <= (152.5 / 255)
    # convert edge pixels to black all else to white
    laplacian[laplacian > (152.5 / 255)] = 0
    laplacian[none_edge_pixels] = 1
    return laplacian


def cartoonify(image, arguments=0):
    laplacian = laplacianAndNormalize(image)
    # picked these numbers from trial and error
    wp = laplacian <= (152.5 / 255)

    # setting these values to white as I'll be subracting
    # so I want the edges to turn black (go to 0) after subtracting
    laplacian[laplacian > (152.5 / 255)] = 1

    # subtract all the image channels from the laplacian (so the edges stand out)
    image = image / 255.0
    for i in range(3):
        image[:, :, i] = image[:, :, i] - laplacian

    # artificially brightening the image (b/c above opperation darkens the image)
    image = image + 0.70

    # renormalizing
    cv2.normalize(
        image, dst=image, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F,
    )

    return image


imagePath = "./images/trump.jpg"
image = cv2.imread(imagePath)

cartoonImage = cartoonify(image)
pencilSketchImage = pencilSketch(image)

plt.figure(figsize=[20, 10])
plt.subplot(131)
plt.imshow(image[:, :, ::-1])
plt.subplot(132)
plt.imshow(cartoonImage[:, :, ::-1])
plt.subplot(133)
plt.imshow(pencilSketchImage)

