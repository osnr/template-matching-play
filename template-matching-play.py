import numpy as np

from scipy.signal import fftconvolve
from normxcorr2 import normxcorr2
from myfftconvolve import myfftconvolve

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2 as cv

import time

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

image = plt.imread("screen.png")
templ = plt.imread("template-traffic-lights.png")

image = cv.resize(image, (0, 0), fx=0.5, fy=0.5)
templ = cv.resize(templ, (0, 0), fx=0.5, fy=0.5)

# image = rgb2gray(image)
# templ = rgb2gray(templ)

def impl_cvMatchTemplate():
    return cv.matchTemplate(image, templ, cv.TM_CCOEFF_NORMED)

def impl_normxcorr2():
    return normxcorr2(templ, image, fftconvolve=fftconvolve, mode="same")

def impl_normxcorr2_myfftconvolve():
    return normxcorr2(templ, image, fftconvolve=myfftconvolve, mode="same")

def test(impl):
    start_time = time.time()
    result = impl()
    print("%s [%s sec]" % (impl.__name__, time.time() - start_time))
    peak = np.unravel_index(np.argmax(result), result.shape)

    fig, ax = plt.subplots(1)
    ax.imshow(result)

    rect = patches.Rectangle((peak[1] - templ.shape[1]/2, peak[0] - templ.shape[0]/2),
                             templ.shape[1], templ.shape[0], linewidth=1,
                             edgecolor='r', facecolor="none")
    ax.add_patch(rect)

test(impl_cvMatchTemplate)
plt.show()
