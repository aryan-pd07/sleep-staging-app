from fastapi import FastAPI
import tensorflow as tf
import numpy as np
import os
from pydantic import BaseModel
from typing import List

class InputData(BaseModel):
    eeg_data: List[List[float]]
    eog_data: List[List[float]]

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

@app.post("/predict_batch")
async def predict_batch(data: InputData):
    try:
        eeg_raw = data.eeg_data
        eog_raw = data.eog_data

        if not eeg_raw or not eog_raw:
            return {"error": "Missing input data"}

        X_eeg = preprocess_batch(eeg_raw)
        X_eog = preprocess_batch(eog_raw)

        raw_preds = model.predict(
            {'EEG_input': X_eeg, 'EOG_input': X_eog},
            verbose=0
        )

        indices = np.argmax(raw_preds, axis=1).tolist()
        confidences = (np.max(raw_preds, axis=1) * 100).tolist()

        return {
            "predictions": indices,
            "confidences": confidences
        }
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
