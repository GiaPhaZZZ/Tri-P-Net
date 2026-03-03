import os
import numpy as np
import librosa
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# ==========================
# CONFIG (can be overridden)
# ==========================
DEFAULT_CFG = {
    "sample_rate": 22050,
    "segment_duration": 3,
    "n_segments": 10,
    "n_mels": 128,
    "n_fft": 2048,
    "hop_length": 512,
    "target_frames": 130,
    "normalize": False,
    "random_state": 42,
}

# ==========================
# Utility functions
# ==========================
def compute_logmel(signal, sr, n_fft, hop_length, n_mels, target_frames):
    mel = librosa.feature.melspectrogram( y=signal,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        center=False,
        power=2.0,
    )

    mel_db = librosa.power_to_db(mel, ref=np.max)

    if mel_db.shape[1] < target_frames:
        pad_width = target_frames - mel_db.shape[1]
        mel_db = np.pad(mel_db, ((0, 0), (0, pad_width)), mode="constant")
    else:
        mel_db = mel_db[:, :target_frames]

    mel_db = mel_db[np.newaxis, ...]

    return mel_db.astype(np.float32)


# ==========================
# Dataset processing
# ==========================
def process_gtzan(
    dataset_path,
    output_path,
    cfg=DEFAULT_CFG,
):
    sr = cfg["sample_rate"]
    samples_per_segment = sr * cfg["segment_duration"]

    splits = ["train", "valid", "test"]
    genres = [
        g for g in os.listdir(dataset_path)
        if os.path.isdir(os.path.join(dataset_path, g))
    ]

    # Create folders
    for split in splits:
        for genre in genres:
            os.makedirs(os.path.join(output_path, split, genre), exist_ok=True)

    for genre in genres:
        print(f"\nProcessing genre: {genre}")
        genre_path = os.path.join(dataset_path, genre)
        files = [f for f in os.listdir(genre_path) if f.endswith(".wav")]

        # 70 / 10 / 20 split
        train_files, temp_files = train_test_split(
            files, test_size=0.3, random_state=cfg["random_state"]
        )
        valid_files, test_files = train_test_split(
            temp_files, test_size=2 / 3, random_state=cfg["random_state"]
        )

        split_map = {
            "train": train_files,
            "valid": valid_files,
            "test": test_files,
        }

        for split in splits:
            print(f"  {split} set...")
            for fname in tqdm(split_map[split]):
                path = os.path.join(genre_path, fname)

                try:
                    signal, _ = librosa.load(path, sr=sr)
                except Exception:
                    continue

                if len(signal) < samples_per_segment * cfg["n_segments"]:
                    continue

                for seg_idx in range(cfg["n_segments"]):
                    start = seg_idx * samples_per_segment
                    end = start + samples_per_segment
                    segment_signal = signal[start:end]

                    feature = compute_logmel(
                        signal=segment_signal,
                        sr=sr,
                        n_fft=cfg["n_fft"],
                        hop_length=cfg["hop_length"],
                        n_mels=cfg["n_mels"],
                        target_frames=cfg["target_frames"],
                    )

                    save_name = f"{fname.replace('.wav', '')}_seg{seg_idx}.npy"
                    save_path = os.path.join(output_path, split, genre, save_name)
                    np.save(save_path, feature)

    print("\nDone processing GTZAN (3s Log-Mel only).")