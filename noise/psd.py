import numpy as np

FFT = np.fft.fft    # use numpy fast fourier transform function


def compute_psd(samples, sample_rate, method = "fft"):
    if method == "fft": # compute psd by fourier transform method
        Ns = samples.size
        Fs = sample_rate
        xdft = FFT(samples)
        xdft = xdft[0:np.floor(Ns/2+1)]
        psdx = (1/(Fs*Ns)) * abs(xdft)**2
        psdx[2:-1] = 2*psdx[2:-1]
        freq = np.arange(0, Fs/2, Fs/Ns)
    elif method == "rxx": # compute psd by correlation method
        pass
    else:
        raise ValueError('Not valid psd computation method. Valid methos are "fft" and "rxx" but gets {}'.format(method))
    return psdx, freq