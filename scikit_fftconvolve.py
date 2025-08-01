# Author: Travis Oliphant
# 1999 -- 2002

from scipy import fft as sp_fft
import numpy as np

def fftconvolve_pow2(in1, in2, mode="full"):
    return fftconvolve(in1, in2, mode=mode, use_pow2=True)

fftconvolve_results = None
def fftconvolve(in1, in2, mode="full", use_pow2=False):
    """Convolve two N-dimensional arrays using FFT.

    Convolve `in1` and `in2` using the fast Fourier transform method, with
    the output size determined by the `mode` argument.

    This is generally much faster than `convolve` for large arrays (n > ~500),
    but can be slower when only a few output values are needed, and can only
    output float arrays (int or object array inputs will be cast to float).

    As of v0.19, `convolve` automatically chooses this method or the direct
    method based on an estimation of which is faster.

    Parameters
    ----------
    in1 : array_like
        First input.
    in2 : array_like
        Second input. Should have the same number of dimensions as `in1`.
    mode : str {'full', 'valid', 'same'}, optional
        A string indicating the size of the output:

        ``full``
           The output is the full discrete linear convolution
           of the inputs. (Default)
        ``valid``
           The output consists only of those elements that do not
           rely on the zero-padding. In 'valid' mode, either `in1` or `in2`
           must be at least as large as the other in every dimension.
        ``same``
           The output is the same size as `in1`, centered
           with respect to the 'full' output.
    axes : int or array_like of ints or None, optional
        Axes over which to compute the convolution.
        The default is over all axes.

    Returns
    -------
    out : array
        An N-dimensional array containing a subset of the discrete linear
        convolution of `in1` with `in2`.

    See Also
    --------
    convolve : Uses the direct convolution or FFT convolution algorithm
               depending on which is faster.
    oaconvolve : Uses the overlap-add method to do convolution, which is
                 generally faster when the input arrays are large and
                 significantly different in size.

    Examples
    --------
    Autocorrelation of white noise is an impulse.

    >>> import numpy as np
    >>> from scipy import signal
    >>> rng = np.random.default_rng()
    >>> sig = rng.standard_normal(1000)
    >>> autocorr = signal.fftconvolve(sig, sig[::-1], mode='full')

    >>> import matplotlib.pyplot as plt
    >>> fig, (ax_orig, ax_mag) = plt.subplots(2, 1)
    >>> ax_orig.plot(sig)
    >>> ax_orig.set_title('White noise')
    >>> ax_mag.plot(np.arange(-len(sig)+1,len(sig)), autocorr)
    >>> ax_mag.set_title('Autocorrelation')
    >>> fig.tight_layout()
    >>> fig.show()

    Gaussian blur implemented using FFT convolution.  Notice the dark borders
    around the image, due to the zero-padding beyond its boundaries.
    The `convolve2d` function allows for other types of image boundaries,
    but is far slower.

    >>> from scipy import datasets
    >>> face = datasets.face(gray=True)
    >>> kernel = np.outer(signal.windows.gaussian(70, 8),
    ...                   signal.windows.gaussian(70, 8))
    >>> blurred = signal.fftconvolve(face, kernel, mode='same')

    >>> fig, (ax_orig, ax_kernel, ax_blurred) = plt.subplots(3, 1,
    ...                                                      figsize=(6, 15))
    >>> ax_orig.imshow(face, cmap='gray')
    >>> ax_orig.set_title('Original')
    >>> ax_orig.set_axis_off()
    >>> ax_kernel.imshow(kernel, cmap='gray')
    >>> ax_kernel.set_title('Gaussian kernel')
    >>> ax_kernel.set_axis_off()
    >>> ax_blurred.imshow(blurred, cmap='gray')
    >>> ax_blurred.set_title('Blurred')
    >>> ax_blurred.set_axis_off()
    >>> fig.show()

    """
    assert(mode == 'same')

    in1 = np.asarray(in1)
    in2 = np.asarray(in2)

    if in1.ndim == in2.ndim == 0:  # scalar inputs
        return in1 * in2
    elif in1.ndim != in2.ndim:
        raise ValueError("in1 and in2 should have the same dimensionality")
    elif in1.size == 0 or in2.size == 0:  # empty arrays
        return np.array([])

    shape = [in1.shape[0] + in2.shape[0] - 1,
             in1.shape[1] + in2.shape[1] - 1]

    ##############

    print("++++++++++++++")
    
    # Speed up FFT by padding to optimal size.
    print("shape", shape)
    fshape_pow2 = [
        (2 ** np.ceil(np.log2(shape[0]))).astype(int),
        (2 ** np.ceil(np.log2(shape[1]))).astype(int)
    ]
    print("fshape_pow2", fshape_pow2)
    fshape_nextfast = [
        sp_fft.next_fast_len(shape[0], True),
        sp_fft.next_fast_len(shape[1], True)
    ]
    print("fshape_nextfast", fshape_nextfast)

    if use_pow2:
        fshape = fshape_pow2
    else:
        fshape = fshape_nextfast
    print("fshape", fshape)

    global fftconvolve_results

    sp1 = sp_fft.rfft2(in1, fshape)
    print("rfft2 in1->sp1: ", in1.shape, sp1.shape, sp1.dtype)
    fftconvolve_results.append(sp1)
    sp2 = sp_fft.rfft2(in2, fshape)
    print("rfft2 in2->sp2: ", in2.shape, sp2.shape, sp2.dtype)
    fftconvolve_results.append(sp2)

    ret = sp_fft.irfft2(sp1 * sp2, fshape)
    print("irfft2 (sp1*sp2) -> ret: ", (sp1 * sp2).shape, ret.shape, ret.dtype)

    # Because of how irfft2 works, ret has been scaled by 1/n, where n
    # is the number of data points. Now reverse the scaling:
    n = np.product(fshape)
    ret = ret * n
    # Now apply a scaling by the cropped number of points instead:
    n_cropped = shape[0] * shape[1]
    ret = ret * (1.0 / n_cropped)

    ret = ret[:shape[0], :shape[1]]
    # save_ret_image(ret)

    ##############

    # Return the center newshape portion of the array.
    newshape = np.asarray(in1.shape)
    currshape = np.array(ret.shape)
    startind = (currshape - newshape) // 2
    endind = startind + newshape
    myslice = [slice(startind[k], endind[k]) for k in range(len(endind))]
    
    ret = ret[tuple(myslice)].copy()
    # save_ret_image(ret)
    print("ret.shape=", ret.shape, "sum(ret)=", np.sum(ret), "mean(ret)=", np.mean(ret))

    # if len(fftconvolve_results) == 0:
    #     # TODO: spit ret and inputs out into images somewhere so we
    #     # can isolatedly test them
    #     # this first call shoudl be `fftconvolve(image, ar.conj(), mode=mode)`
    #     # save in1 to in1-UUID file
    #     # save in2 to in2-UUID file
    #     # save ret to ret-UUID file
    fftconvolve_results.append(ret)
    return ret

import uuid
import matplotlib.pyplot as plt

def save_ret_image(ret):
    filename = f"{uuid.uuid4()}.png"
    plt.figure(figsize=(10, 10))
    plt.imshow(ret, cmap='viridis')
    plt.colorbar()
    plt.title('FFT Convolution Result')
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"Image saved as: {filename}")
    return filename
