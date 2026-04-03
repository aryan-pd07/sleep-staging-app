from fastapi import FastAPI
import tensorflow as tf
import numpy as np
import os
import mne
import tempfile
import requests as http_requests
from pydantic import BaseModel
from typing import List, Optional

class InputData(BaseModel):
    eeg_data: Optional[List[List[float]]] = None
    eog_data: Optional[List[List[float]]] = None
    file_url: Optional[str] = None
    eeg_channel: Optional[str] = "EEG Fpz-Cz"
    eog_channel: Optional[str] = "EOG horizontal"

app = FastAPI(title="NeuroSleep AI Backend")
MODEL_PATH = 'model/best_cnn_lstm_model.keras'

try:
    if os.path.exists(MODEL_PATH):
        model = tf.keras.models.load_model(MODEL_PATH)
        print("✅ Model loaded successfully.")
    else:
        print(f"❌ Model file not found at {MODEL_PATH}")
except Exception as e:
    print(f"❌ Error loading model: {e}")

def preprocess_batch(data_list):
    processed = []
    for seg in data_list:
        seg = np.array(seg)
        if np.max(np.abs(seg)) < 0.01:
            seg *= 1e6
        if np.std(seg) > 0:
            seg = (seg - np.mean(seg)) / np.std(seg)
        processed.append(seg.reshape(3000, 1))
    return np.array(processed)

def load_edf_from_url(url, eeg_channel, eog_channel):
    response = http_requests.get(url, timeout=60)
    with tempfile.NamedTemporaryFile(delete=False, suffix='.edf') as tfile:
        tfile.write(response.content)
        tfile_path = tfile.name
    raw = mne.io.read_raw_edf(tfile_path, preload=True, verbose=False)
    os.unlink(tfile_path)
    if raw.info['sfreq'] != 100:
        raw.resample(100, npad="auto")
    eeg = raw.get_data(picks=[eeg_channel])[0]
    eog = raw.get_data(picks=[eog_channel])[0]
    return eeg, eog

@app.post("/predict_batch")
async def predict_batch(data: InputData):
    try:
        if data.file_url:
            eeg, eog = load_edf_from_url(data.file_url, data.eeg_channel, data.eog_channel)
            num_epochs = min(12, len(eeg) // 3000)
            eeg_raw = [eeg[i*3000:(i+1)*3000].tolist() for i in range(num_epochs)]
            eog_raw = [eog[i*3000:(i+1)*3000].tolist() for i in range(num_epochs)]
        elif data.eeg_data and data.eog_data:
            eeg_raw = data.eeg_data
            eog_raw = data.eog_data
        else:
            return {"error": "No input data provided"}

        X_eeg = preprocess_batch(eeg_raw)
        X_eog = preprocess_batch(eog_raw)
        raw_preds = model.predict({'EEG_input': X_eeg, 'EOG_input': X_eog}, verbose=0)
        indices = np.argmax(raw_preds, axis=1).tolist()
        confidences = (np.max(raw_preds, axis=1) * 100).tolist()
        return {"predictions": indices, "confidences": confidences}
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
