import numpy as np

# let's try reimplementing this
# https://stackoverflow.com/questions/40703751/using-fourier-transforms-to-do-convolution
def myfftconvolve(f, g, mode):
    size = np.array(f.shape) + np.array(g.shape) - 1
    fsize = 2 ** np.ceil(np.log2(size)).astype(int)
    f_ = np.fft.fft2(f, fsize)
    g_ = np.fft.fft2(g, fsize)
    FG = f_ * g_
    return np.real(np.fft.ifft2(FG))
