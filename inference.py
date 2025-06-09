import torch
import torchaudio
import argparse
import os
import torch.nn.functional as F

from cnn import CNNNetwork
from train import SAMPLE_RATE, NUM_SAMPLES

# Danh sách thể loại nhạc GTZAN
class_mapping = [
    "blues", "classical", "country", "disco", "hiphop",
    "jazz","metal", "pop", "reggae", "rock",
]

def predict(model, input_tensor, class_mapping):
    model.eval()
    with torch.no_grad():
        predictions = model(input_tensor)
        probabilities = F.softmax(predictions[0], dim=0)  # softmax để chuyển thành xác suất
        predicted_index = probabilities.argmax().item()
        predicted_label = class_mapping[predicted_index]
    return predicted_label, probabilities

def preprocess_audio(file_path, transformation, target_sample_rate, num_samples, device):
    signal, sr = torchaudio.load(file_path)
    signal = signal.to(device)

    if sr != target_sample_rate:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sample_rate).to(device)
        signal = resampler(signal)

    if signal.shape[0] > 1:
        signal = torch.mean(signal, dim=0, keepdim=True)

    if signal.shape[1] > num_samples:
        signal = signal[:, :num_samples]
    if signal.shape[1] < num_samples:
        padding = num_samples - signal.shape[1]
        signal = torch.nn.functional.pad(signal, (0, padding))

    spec = transformation(signal)
    return spec

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dự đoán thể loại nhạc từ file .wav")
    parser.add_argument("file_path", type=str, help="Đường dẫn tới file .wav")
    args = parser.parse_args()

    if not os.path.exists(args.file_path):
        print(" File không tồn tại.")
        exit()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    cnn = CNNNetwork().to(device)
    state_dict = torch.load("cnnnet.pth", map_location=device)
    cnn.load_state_dict(state_dict)

    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=1024,
        hop_length=512,
        n_mels=64
    ).to(device)

    input_tensor = preprocess_audio(args.file_path, mel_spectrogram, SAMPLE_RATE, NUM_SAMPLES, device)
    input_tensor = input_tensor.unsqueeze(0)

    predicted_label, probabilities = predict(cnn, input_tensor, class_mapping)

    print(f"\n Thể loại: '{predicted_label}'\n")
    print(" Xác suất từng thể loại:")
    for label, prob in zip(class_mapping, probabilities):
        print(f"  {label:12s}: {prob.item() * 100:.2f}%")
