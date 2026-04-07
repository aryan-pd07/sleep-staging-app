"""
utils.py — NeuroSleep AI
Helper functions for signal processing, inference, and data loading.
"""

import numpy as np
import pandas as pd
import mne
import tempfile
import os


# ─────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────

CLASS_LABELS = {0: "Wake", 1: "N1", 2: "N2", 3: "N3", 4: "REM"}

HYPNOGRAM_MAP = {"Wake": 4, "REM": 3, "N1": 2, "N2": 1, "N3": 0}

STAGE_COLORS = {
    "Wake": "#e74c3c",
    "REM": "#3498db",
    "N1": "#f1c40f",
    "N2": "#95a5a6",
    "N3": "#2ecc71"
}

LABEL_MAP = {
    "sleep stage w": 0, "wake": 0,
    "sleep stage 1": 1, "n1": 1, "sleep stage n1": 1,
    "sleep stage 2": 2, "n2": 2, "sleep stage n2": 2,
    "sleep stage 3": 3, "n3": 3, "sleep stage n3": 3,
    "sleep stage 4": 3,
    "sleep stage r": 4, "rem": 4, "sleep stage rem": 4,
    "movement": 0, "unknown": 0
}


# ─────────────────────────────────────────
# SIGNAL PREPROCESSING
# ─────────────────────────────────────────

def preprocess_batch(data_list):
    """
    Standardizes and reshapes a list of 30-second epochs for CNN-LSTM input.

    Args:
        data_list: List of epochs, each with 3000 samples (30s @ 100Hz)

    Returns:
        np.ndarray of shape (N, 3000, 1)
    """
    processed = []
    for seg in data_list:
        seg = np.array(seg, dtype=np.float32)
        # Scale to microvolts if signal amplitude is very low
        if np.max(np.abs(seg)) < 0.01:
            seg *= 1e6
        # Z-score normalization
        if np.std(seg) > 0:
            seg = (seg - np.mean(seg)) / np.std(seg)
        processed.append(seg.reshape(3000, 1))
    return np.array(processed)


def smooth_predictions(pred, window_size=3):
    """
    Applies majority-vote smoothing to reduce noisy stage transitions.

    Args:
        pred: np.ndarray of integer class predictions
        window_size: size of the smoothing window (default: 3 epochs)

    Returns:
        Smoothed np.ndarray of predictions
    """
    smoothed = pred.copy()
    half_w = window_size // 2
    for i in range(len(pred)):
        start = max(0, i - half_w)
        end = min(len(pred), i + half_w + 1)
        smoothed[i] = np.bincount(pred[start:end]).argmax()
    return smoothed


# ─────────────────────────────────────────
# MODEL INFERENCE
# ─────────────────────────────────────────

def run_inference(model, eeg_epochs, eog_epochs):
    """
    Runs the CNN-LSTM model on a batch of EEG and EOG epochs.

    Args:
        model: Loaded Keras model
        eeg_epochs: List of EEG epoch arrays (each 3000 samples)
        eog_epochs: List of EOG epoch arrays (each 3000 samples)

    Returns:
        Tuple of (predictions, confidences) as np.ndarrays
    """
    if model is None:
        return None, None

    X_eeg = preprocess_batch(eeg_epochs)
    X_eog = preprocess_batch(eog_epochs)

    raw_preds = model.predict(
        {'EEG_input': X_eeg, 'EOG_input': X_eog},
        verbose=0
    )

    indices = np.argmax(raw_preds, axis=1)
    confidences = np.max(raw_preds, axis=1) * 100
    return indices, confidences


# ─────────────────────────────────────────
# EDF FILE LOADING
# ─────────────────────────────────────────

def load_edf_file(uploaded_file):
    """
    Reads an uploaded EDF file and returns the raw MNE object.

    Args:
        uploaded_file: Streamlit UploadedFile object

    Returns:
        Tuple of (raw, error_message)
    """
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.edf') as tfile:
            tfile.write(uploaded_file.getvalue())
            tfile_path = tfile.name

        raw = mne.io.read_raw_edf(tfile_path, preload=True, verbose=False)
        os.unlink(tfile_path)

        if raw.info['sfreq'] != 100:
            raw.resample(100, npad="auto")

        return raw, None
    except Exception as e:
        return None, str(e)


# ─────────────────────────────────────────
# GROUND TRUTH LOADER
# ─────────────────────────────────────────

def load_ground_truth(file_obj, num_epochs, start_epoch_idx):
    """
    Parses a hypnogram file and returns per-epoch integer labels.

    Supports EDF annotations, CSV, and TXT formats.

    Args:
        file_obj: Streamlit UploadedFile object
        num_epochs: Number of epochs to extract
        start_epoch_idx: Starting epoch index

    Returns:
        np.ndarray of integer labels, or None on failure
    """
    try:
        fname = file_obj.name.lower()

        if fname.endswith('.edf'):
            with tempfile.NamedTemporaryFile(delete=False, suffix='.edf') as tfile:
                tfile.write(file_obj.getvalue())
                tpath = tfile.name
            annot = mne.read_annotations(tpath)
            os.unlink(tpath)

            labels = []
            for i in range(num_epochs):
                midpoint = (start_epoch_idx + i) * 30 + 15
                found_label = "unknown"
                for desc, onset, duration in zip(annot.description, annot.onset, annot.duration):
                    if onset <= midpoint < (onset + duration):
                        found_label = desc.lower()
                        break
                labels.append(LABEL_MAP.get(found_label, -1))
            return np.array(labels)

        elif fname.endswith('.txt') or fname.endswith('.csv'):
            df = pd.read_csv(file_obj, header=None)
            full_labels = df.iloc[:, 0].astype(str).str.lower().map(LABEL_MAP).values
            end_idx = start_epoch_idx + num_epochs
            if end_idx > len(full_labels):
                return None
            return full_labels[start_epoch_idx:end_idx]

    except Exception as e:
        print(f"GT Error: {e}")
        return None


# ─────────────────────────────────────────
# DEMO DATA
# ─────────────────────────────────────────

def get_dummy_data():
    """
    Generates 30 minutes of synthetic EEG and EOG signals for demo purposes.

    Returns:
        Tuple of (eeg, eog) as np.ndarrays of shape (180000,)
    """
    return (
        np.random.normal(0, 1, (180000,)).astype(np.float32),
        np.random.normal(0, 1, (180000,)).astype(np.float32)
    )
