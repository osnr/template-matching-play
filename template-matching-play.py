import numpy as np

from normxcorr2 import normxcorr2
from scipy.signal import fftconvolve
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

image = rgb2gray(image)
templ = rgb2gray(templ)

def impl_cvMatchTemplate():
    return cv.matchTemplate(np.float32(image), np.float32(templ), cv.TM_CCOEFF_NORMED)

def impl_normxcorr2():
    return normxcorr2(templ, image, fftconvolve=fftconvolve, mode="same")

def impl_normxcorr2_myfftconvolve():
    return normxcorr2(templ, image, fftconvolve=myfftconvolve, mode="same")

results = []
def run(impl):
    start_time = time.time()
    result = impl()
    exec_time = time.time() - start_time
    print("%s [%s sec]" % (impl.__name__, exec_time))
    print(result.dtype, result.shape)
    results.append((impl.__name__, result, exec_time))

    print()

def done():
    fig, axs = plt.subplots(len(results), 2)
    for i, (impl_name, result, exec_time) in enumerate(results):
        peak = np.unravel_index(np.argmax(result), result.shape)

        axs[i, 0].set_title("%s [%.03f sec]" % (impl_name, exec_time))
        axs[i, 0].imshow(result)

        rect = patches.Rectangle((peak[1] - templ.shape[1]/2, peak[0] - templ.shape[0]/2),
                                 templ.shape[1], templ.shape[0],
                                 linewidth=1, edgecolor='r', facecolor="none")
        axs[i, 0].add_patch(rect)

    plt.tight_layout()
    plt.show()

run(impl_cvMatchTemplate)
run(impl_normxcorr2)
run(impl_normxcorr2_myfftconvolve)

done()


