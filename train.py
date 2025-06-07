import torch
import torchaudio
from torch import nn
from torch.utils.data import DataLoader
import numpy as np

from musicdataset import MusicGenreDataset
from cnn import CNNNetwork

BATCH_SIZE = 64
EPOCHS = 20
LEARNING_RATE = 0.001

AUDIO_DIR = "C:\\Users\\dohuu\\Downloads\\archive\\Data\\genres_original"
SAMPLE_RATE = 22050
NUM_SAMPLES = SAMPLE_RATE * 20

def create_data_loader(train_data, batch_size):
    labels = [label for _, label in train_data]
    class_sample_count = np.bincount(labels)
    class_weights = 1. / class_sample_count
    sample_weights = [class_weights[label] for label in labels]
    sampler = torch.utils.data.WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
    return DataLoader(train_data, batch_size=batch_size, sampler=sampler)

def train_single_epoch(model, data_loader, loss_fn, optimiser, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for input, target in data_loader:
        input, target = input.to(device), target.to(device)

        prediction = model(input)
        loss = loss_fn(prediction, target)

        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

        running_loss += loss.item() * input.size(0)
        _, predicted = torch.max(prediction, 1)
        correct += (predicted == target).sum().item()
        total += target.size(0)

    epoch_loss = running_loss / total
    accuracy = correct / total * 100
    print(f"Loss: {epoch_loss:.4f}, Accuracy: {accuracy:.2f}%")

def train(model, data_loader, loss_fn, optimiser, device, epochs):
    for i in range(epochs):
        print(f"Epoch {i+1}/{epochs}")
        train_single_epoch(model, data_loader, loss_fn, optimiser, device)
        print("---------------------------")
    print("Finished training")

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device}")

    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=1024,
        hop_length=512,
        n_mels=64
    )

    dataset = MusicGenreDataset(
        AUDIO_DIR, mel_spectrogram, SAMPLE_RATE, NUM_SAMPLES, device
    )

    train_loader = create_data_loader(dataset, BATCH_SIZE)

    model = CNNNetwork().to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimiser = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train(model, train_loader, loss_fn, optimiser, device, EPOCHS)

    torch.save(model.state_dict(), "cnnnet.pth")
    print("Trained CNN model saved at cnnnet.pth")
