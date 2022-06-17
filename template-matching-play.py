#!/usr/bin/env python3

import numpy as np

from normxcorr2 import normxcorr2
from normxcorr2_fewer_ffts import normxcorr2_fewer_ffts
from scipy.signal import fftconvolve
from myfftconvolve import myfftconvolve

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2 as cv
from skimage_template import match_template

import sys
import time
from copy import copy

if len(sys.argv) == 1:
    IMAGE_FILE, TEMPL_FILE = "screen.png", "template-traffic-lights.png"
elif len(sys.argv) == 3:
    IMAGE_FILE, TEMPL_FILE = sys.argv[1], sys.argv[2]
else:
    sys.stderr.write("%s IMAGE_FILE TEMPLATE_FILE\n" % (sys.argv[0]))
    exit(1)

image = plt.imread(IMAGE_FILE)
templ = plt.imread(TEMPL_FILE)

image = cv.resize(image, (0, 0), fx=0.5, fy=0.5)
templ = cv.resize(templ, (0, 0), fx=0.5, fy=0.5)

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

image = rgb2gray(image)
templ = rgb2gray(templ)

print("image", image.shape)
print("templ", templ.shape)
print()

def impl_cvMatchTemplate():
    return cv.matchTemplate(np.float32(image), np.float32(templ), cv.TM_CCOEFF_NORMED)

def impl_skimage():
    return match_template(np.float32(image), np.float32(templ))

def impl_normxcorr2():
    return normxcorr2(templ, image, fftconvolve=fftconvolve, mode="same")

def impl_normxcorr2_fewer_ffts():
    return normxcorr2_fewer_ffts(templ, image, mode="same")

def impl_normxcorr2_myfftconvolve():
    return normxcorr2(templ, image, fftconvolve=myfftconvolve, mode="same")

results = []
def run(impl):
    start_time = time.time()
    result = impl()
    elapsed = time.time() - start_time
    print("%s [%s sec]" % (impl.__name__, elapsed))
    print(result.dtype, result.shape)
    results.append((impl.__name__, result, elapsed))

    print()

def done():
    fig, axs = plt.subplots(len(results), 2)
    for i, (impl_name, result, elapsed) in enumerate(results):
        axs[i, 0].imshow(result)
        axs[i, 1].imshow(image)

        peaks = np.argwhere(result > 0.9)
        axs[i, 0].set_title("%s [%d matches] [%.03f sec]" % (impl_name, len(peaks), elapsed))
        for peak in peaks:
            rect = patches.Rectangle((peak[1] - templ.shape[1]/2, peak[0] - templ.shape[0]/2),
                                     templ.shape[1], templ.shape[0],
                                     linewidth=1, edgecolor='r', facecolor="none")
            axs[i, 0].add_patch(copy(rect))
            axs[i, 1].add_patch(copy(rect))

    plt.tight_layout()
    plt.show()

run(impl_cvMatchTemplate)
run(impl_skimage)
run(impl_normxcorr2)
# run(impl_normxcorr2_fewer_ffts)
run(impl_normxcorr2_myfftconvolve)

done()
