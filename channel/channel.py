import numpy as np
class Channel:
    def __init__(self, chan_type = "awgn", snr_db = 10):
        self.chan_type = chan_type
        self.snr_db = snr_db

    
    def gaussian_noise(self, x, sigma):
        noise_real = np.random.normal(0, sigma, x.shape)
        noise_imag = np.random.normal(0, sigma, x.shape)
        return x + (noise_real + 1j * noise_imag)
    
    def __call__(self, x):
        avg_power = np.mean(np.abs(x) ** 2)
        x = x / np.sqrt(np.mean(np.abs(x)**2))
        sigma = np.sqrt(1 / (2 * (10 ** (self.snr_db / 10))))
        if self.chan_type == "awgn":
            return self.gaussian_noise(x, sigma)
        if self.chan_type == "rayleigh":
            H_real = np.random.normal(0, np.sqrt(1/2))
            H_imag = np.random.normal(0, np.sqrt(1/2))
            H = H_real + 1j * H_imag
            y = x * H
            y = self.gaussian_noise(y, sigma)
            return y / H
        else:
            raise ValueError(f"Unsupported channel type {self.chan_type}")
        