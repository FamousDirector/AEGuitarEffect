import torchaudio
import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader


class AudioSignalDataset(Dataset):
    def __init__(self, input_directory, sample_length=1500, sample_shift=1000):
        """

        :param input_directory: directory to pull audio from
        :param sample_length: how long to make the samples
        :param sample_shift: how much to shift the samples every time
        """

        self.sample_rates = []
        self.complete_waveforms = []
        self.sample_waveforms = []

        for file in os.listdir(input_directory):
            if file.endswith(".wav"):
                waveform, sample_rate = torchaudio.load(os.path.join(input_directory, file))
                self.sample_rates.append(sample_rate)
                self.complete_waveforms.append(waveform[0])

        for w in self.complete_waveforms:
            i = 0
            while len(w) > (i + sample_length):
                self.sample_waveforms.append(w[i:i+sample_length].unsqueeze(0))
                i += sample_shift

    def __len__(self):
        return len(self.sample_waveforms)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = self.sample_waveforms[idx]

        return sample
