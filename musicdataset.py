import os
import torch
from torch.utils.data import Dataset
import torchaudio

class MusicGenreDataset(Dataset):
    def __init__(self, audio_dir, transformation, target_sample_rate, num_samples, device):
        self.audio_dir = audio_dir
        self.device = device
        self.transformation = transformation.to(self.device)
        self.target_sample_rate = target_sample_rate
        self.num_samples = num_samples

        self.classes = sorted(os.listdir(audio_dir))
        self.audio_paths = []
        self.labels = []

        for label_idx, genre in enumerate(self.classes):
            genre_path = os.path.join(audio_dir, genre)
            for file in os.listdir(genre_path):
                if file.endswith(".wav"):
                    self.audio_paths.append(os.path.join(genre_path, file))
                    self.labels.append(label_idx)

    def __len__(self):
        return len(self.audio_paths)

    def __getitem__(self, index):
        audio_path = self.audio_paths[index]
        label = self.labels[index]

        try:
            signal, sr = torchaudio.load(audio_path)
        except Exception as e:
            return self.__getitem__((index + 1) % len(self.audio_paths))

        signal = signal.to(self.device)
        signal = self._resample_if_necessary(signal, sr)
        signal = self._mix_down_if_necessary(signal)
        signal = self._cut_if_necessary(signal)
        signal = self._right_pad_if_necessary(signal)
        signal = self.transformation(signal)
        return signal, label

    def _resample_if_necessary(self, signal, sr):
        if sr != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.target_sample_rate).to(self.device)
            signal = resampler(signal)
        return signal

    def _mix_down_if_necessary(self, signal):
        if signal.shape[0] > 1:
            signal = torch.mean(signal, dim=0, keepdim=True)
        return signal

    def _cut_if_necessary(self, signal):
        if signal.shape[1] > self.num_samples:
            signal = signal[:, :self.num_samples]
        return signal

    def _right_pad_if_necessary(self, signal):
        if signal.shape[1] < self.num_samples:
            num_missing = self.num_samples - signal.shape[1]
            signal = torch.nn.functional.pad(signal, (0, num_missing))
        return signal
