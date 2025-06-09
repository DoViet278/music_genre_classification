import streamlit as st
import torch
import torchaudio
import torch.nn.functional as F
import os
import matplotlib.pyplot as plt
from cnn import CNNNetwork
from train import SAMPLE_RATE, NUM_SAMPLES

# --- Cấu hình ---
class_mapping = [
    "blues", "classical", "country", "disco", "hiphop",
    "jazz", "metal", "pop", "reggae", "rock"
]

device = "cuda" if torch.cuda.is_available() else "cpu"

# --- Load model ---
cnn = CNNNetwork().to(device)
state_dict = torch.load("cnnnet.pth", map_location=device)
cnn.load_state_dict(state_dict)
cnn.eval()

# --- MelSpectrogram transform ---
mel_spectrogram = torchaudio.transforms.MelSpectrogram(
    sample_rate=SAMPLE_RATE,
    n_fft=1024,
    hop_length=512,
    n_mels=64
).to(device)

# --- Dự đoán ---
def predict(model, input_tensor, class_mapping):
    with torch.no_grad():
        predictions = model(input_tensor)
        probabilities = F.softmax(predictions[0], dim=0)
        predicted_index = probabilities.argmax().item()
        predicted = class_mapping[predicted_index]
    return predicted, probabilities.cpu().numpy()

# --- Tiền xử lý ---
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
    elif signal.shape[1] < num_samples:
        padding = num_samples - signal.shape[1]
        signal = torch.nn.functional.pad(signal, (0, padding))

    spec = transformation(signal)
    return spec

# --- Giao diện Streamlit ---
st.set_page_config(page_title="Dự đoán thể loại nhạc", layout="centered")
st.title("🎵 Dự đoán thể loại nhạc từ file WAV, MP3")

uploaded_file = st.file_uploader("Tải lên file ", type=["wav", "mp3"])

if uploaded_file is not None:
    st.audio(uploaded_file, format="audio/wav")

    # Lưu file tạm thời
    with open("temp.wav", "wb") as f:
        f.write(uploaded_file.read())

    try:
        input_tensor = preprocess_audio("temp.wav", mel_spectrogram, SAMPLE_RATE, NUM_SAMPLES, device)
        input_tensor = input_tensor.unsqueeze(0)
        predicted_class, probs = predict(cnn, input_tensor, class_mapping)

        st.success(f"🎧 Thể loại dự đoán: **{predicted_class}**")

        st.subheader("Xác suất từng thể loại:")
        labels = class_mapping
        probs_percent = (probs * 100).round(2)

        fig, ax = plt.subplots()
        bars = ax.barh(labels, probs_percent, color='skyblue')
        ax.set_xlabel("Xác suất (%)")
        ax.set_xlim(0, 100)

        # Hiển thị giá trị phần trăm trên mỗi thanh
        for bar, prob in zip(bars, probs_percent):
            ax.text(prob + 1, bar.get_y() + bar.get_height()/2, f"{prob:.2f}%", va='center')

        st.pyplot(fig)

    except Exception as e:
        st.error(f"Lỗi khi dự đoán: {e}")

    # Xoá file tạm
    os.remove("temp.wav")
