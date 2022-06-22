########################################################################################
# modified by Omar Rizwan (2022) to add fftconvolve function parameter, reduce use of fftconvolve #
#                                                                                      #
# Author: Ujash Joshi, University of Toronto, 2017                                     #
# Based on Octave implementation by: Benjamin Eltzner, 2014 <b.eltzner@gmx.de>         #
# Octave/Matlab normxcorr2 implementation in python 3.5                                #
# Details:                                                                             #
# Normalized cross-correlation. Similiar results upto 3 significant digits.            #
# https://github.com/Sabrewarrior/normxcorr2-python/tree/master/normxcorr2.py          #
# http://lordsabre.blogspot.ca/2017/09/matlab-normxcorr2-implemented-in-python.html    #
########################################################################################

import numpy as np
from scipy.signal import fftconvolve

def lsum(image, window_shape):

    window_sum = np.cumsum(image, axis=0)
    window_sum = (window_sum[window_shape[0]:-1]
                  - window_sum[:-window_shape[0] - 1])

    window_sum = np.cumsum(window_sum, axis=1)
    window_sum = (window_sum[:, window_shape[1]:-1]
                  - window_sum[:, :-window_shape[1] - 1])

    return window_sum


def normxcorr2_fewer_ffts(template, image, mode="full"):
    """
    Input arrays should be floating point numbers.
    :param template: N-D array, of template or filter you are using for cross-correlation.
    Must be less or equal dimensions to image.
    Length of each dimension must be less than length of image.
    :param image: N-D array
    :param mode: Options, "full", "valid", "same"
    full (Default): The output of fftconvolve is the full discrete linear convolution of the inputs. 
    Output size will be image size + 1/2 template size in each dimension.
    valid: The output consists only of those elements that do not rely on the zero-padding.
    same: The output is the same size as image, centered with respect to the ‘full’ output.
    :return: N-D array of same dimensions as image. Size depends on mode parameter.
    """

    # If this happens, it is probably a mistake
    if np.ndim(template) > np.ndim(image) or \
            len([i for i in range(np.ndim(template)) if template.shape[i] > image.shape[i]]) > 0:
        print("normxcorr2: TEMPLATE larger than IMG. Arguments may be swapped.")

    print('image shape', image.shape)
    print('template shape', template.shape)

    template = template - np.mean(template)
    image = image - np.mean(image)

    # Faster to flip up down and left right then use fftconvolve instead of scipy's correlate
    ar = np.flipud(np.fliplr(template))
    out = fftconvolve(image, ar.conj(), mode=mode)


    # FIXME: instead of the below, use numpy csum
    if True:
        a1 = np.ones(template.shape)
        minuend = fftconvolve(np.square(image), a1, mode=mode)
        subtrahend = np.square(fftconvolve(image, a1, mode=mode)) / np.prod(template.shape)
        print('minuend', minuend.shape)
        print('subtrahend', subtrahend.shape)
        # image = minuend - subtrahend

    if True:
        minuend2 = lsum(np.square(image), template.shape)
        subtrahend2 = np.square(lsum(image, template.shape)) / np.prod(template.shape)
        print('minuend2', minuend2.shape)
        print('subtrahend2', subtrahend2.shape)
        image = minuend2 - subtrahend2

    # Remove small machine precision errors after subtraction
    image[np.where(image < 0)] = 0

    templateSum = np.sum(np.square(template))
    print("image shape", image.shape)
    out = out / np.pad(np.sqrt(image * templateSum), ((template.shape[0] + 1, 0), (template.shape[1] + 1, 0)), 'edge')

    # Remove any divisions by 0 or very close to 0
    out[np.where(np.logical_not(np.isfinite(out)))] = 0
    
    return out
