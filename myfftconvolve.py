import numpy as np


def myfftconvolve(f, g, mode):  # our impl
    # Pad g to equal size of f. This assumes g is smaller in both dimensions
    # and that the difference between f and g dimensions are even
    p1, p2 = (np.r_[f.shape]-g.shape)/2

    gpad = np.pad(g, ((np.int(np.floor(p1)), np.int(np.ceil(p1))),
                      (np.int(np.floor(p2)), np.int(np.ceil(p2)))),
                  mode='edge')

    # Shift g to 'center' on top left corner (the 'origin' in an fft)
    gpad = np.fft.ifftshift(gpad)

    # Multiply spectra
    FG = np.fft.fft2(f) * np.conj(np.fft.fft2(gpad))

    return np.real(np.fft.ifft2(FG))
