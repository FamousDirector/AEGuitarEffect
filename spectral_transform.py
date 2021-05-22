from scipy import signal
import numpy as np
import torch


class SpectralTransform:
    def __init__(self, sampling_frequency, segment_length=128, overlap_length=128-1, fft_length=128):
        self.fs = sampling_frequency
        self.nperseg = segment_length
        self.noverlap = overlap_length
        self.nfft = fft_length

    def transform(self, x):
        _, _, Zxx = signal.stft(x, self.fs, nperseg=self.nperseg, noverlap=self.noverlap, nfft=self.nfft)

        if torch.is_tensor(x):
            return torch.tensor(Zxx.real), torch.tensor(Zxx.imag)
        else:
            return Zxx.real, Zxx.imag

    def inverse_transform(self, real, imag):
        if torch.is_tensor(real):
            r = real.detach().numpy()
            i = imag.detach().numpy()
            Zxx = np.imag(r + i * 1j)
        else:
            Zxx = np.imag(real + imag * 1j)

        _, xrec = signal.istft(Zxx, self.fs, nperseg=self.nperseg, noverlap=self.noverlap, nfft=self.nfft)

        if torch.is_tensor(real):
            xrec = torch.tensor(xrec, requires_grad=True)
        return xrec
