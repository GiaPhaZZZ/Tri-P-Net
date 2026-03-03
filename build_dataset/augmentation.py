import torch
import random
import numpy as np

class SpecAugment:
    def __init__(self,
                 time_mask_param=20,
                 freq_mask_param=15,
                 num_time_masks=2,
                 num_freq_masks=2,
                 p=0.8):
        self.time_mask_param = time_mask_param
        self.freq_mask_param = freq_mask_param
        self.num_time_masks = num_time_masks
        self.num_freq_masks = num_freq_masks
        self.p = p

    def __call__(self, spec):
        # spec: (3, 128, 130)

        if random.random() > self.p:
            return spec
        
        spec = spec.clone()

        C, F, T = spec.shape

        # --- Time Masking ---
        for _ in range(self.num_time_masks):
            t = random.randint(0, self.time_mask_param)
            t0 = random.randint(0, max(1, T - t))
            spec[:, :, t0:t0+t] = 0

        # --- Frequency Masking ---
        for _ in range(self.num_freq_masks):
            f = random.randint(0, self.freq_mask_param)
            f0 = random.randint(0, max(1, F - f))
            spec[:, f0:f0+f, :] = 0

        return spec


class SpectrogramAugmentation:
    def __init__(self):
        self.specaugment = SpecAugment()

    def __call__(self, spec):
        # Random time shift
        if random.random() < 0.5:
            shift = random.randint(-10, 10)
            spec = torch.roll(spec, shifts=shift, dims=2)

        # Random gain
        if random.random() < 0.5:
            gain = random.uniform(0.8, 1.2)
            spec = spec * gain

        # Gaussian noise
        if random.random() < 0.5:
            noise = torch.randn_like(spec) * 0.01
            spec = spec + noise

        # SpecAugment
        spec = self.specaugment(spec)

        return spec
