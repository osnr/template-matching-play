import numpy as np

import matplotlib.pyplot as plt

# let's try reimplementing this
# https://stackoverflow.com/questions/40703751/using-fourier-transforms-to-do-convolution
def myfftconvolve(f, g, mode):
    size = np.array(f.shape) + np.array(g.shape) - 1
    fsize = 2 ** np.ceil(np.log2(size)).astype(int)
    f_ = np.fft.fft2(f, fsize)
    g_ = np.fft.fft2(g, fsize)
    FG = f_ * g_
    height, width = f.shape
    return np.real(np.fft.ifft2(FG))[0:height, 0:width]

def myfftconvolve2(f, g, mode):
    # Pad g to equal size of f.
    d1, d2 = (np.r_[f.shape]-g.shape).astype(int)
    # print('pad', ((np.floor(d1/2).astype(int), np.ceil(d1/2).astype(int)),
    #                   (np.floor(d2/2).astype(int), np.ceil(d2/2).astype(int))))
    gpad = np.pad(g, ((np.floor(d1/2).astype(int), np.ceil(d1/2).astype(int)),
                      (np.floor(d2/2).astype(int), np.ceil(d2/2).astype(int))), mode='edge')

    # Shift g to 'center' on top left corner (the 'origin' in an fft)
    # pad = np.fft.ifftshift(gpad)

    # plt.figure(figsize=(8, 6))
    # plt.imshow(gpad, cmap='viridis')
    # plt.colorbar(label='Value')
    # plt.title('Padded Kernel (gpad)')
    # plt.xlabel('X')
    # plt.ylabel('Y')
    # plt.show()

    # Multiply spectra
    FG = np.fft.fft2(f) * np.conj(np.fft.fft2(gpad))
    return np.real(np.fft.ifft2(FG))
