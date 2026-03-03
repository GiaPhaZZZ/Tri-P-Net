import os
import numpy as np
import torch
from torch.utils.data import Dataset

from build_dataset.augmentation import SpectrogramAugmentation

def get_samples(root_dir, split_name):
    
    split_path = os.path.join(root_dir, split_name)
    samples = []

    # Get genres (folders like 'blues', 'classical', etc.)
    genres = sorted([
        g for g in os.listdir(split_path)
        if os.path.isdir(os.path.join(split_path, g))
    ])
    class_map = {genre: idx for idx, genre in enumerate(genres)}

    for genre in genres:
        genre_path = os.path.join(split_path, genre)
        
        # Look for .npy files directly inside the genre folder
        for file_name in os.listdir(genre_path):
            if file_name.endswith(".npy"):
                npy_path = os.path.join(genre_path, file_name)
                samples.append((npy_path, class_map[genre]))

    return samples, class_map


class MelNPYDataset(Dataset):
    def __init__(self, samples, train=True):
        self.samples = samples
        self.train = train
        self.augment = SpectrogramAugmentation() if train else None

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        npy_path, label = self.samples[idx]

        # Load numpy array
        mel = np.load(npy_path)

        # Convert once to tensor
        mel = torch.from_numpy(mel).float()

        if mel.dim() == 2:
            mel = mel.unsqueeze(0)

        # Apply augmentation only in train mode
        if self.train and self.augment is not None:
            mel = self.augment(mel)

        # Compute complementary mel AFTER augmentation
        mel_comp = mel.max() - mel

        return mel, mel_comp, label