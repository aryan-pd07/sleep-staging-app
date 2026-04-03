from fastapi import FastAPI, Request
import tensorflow as tf
import numpy as np
import os
from pydantic import BaseModel
from typing import List
class InputData(BaseModel):
    eeg_data: List[float]
    eog_data: List[float]

app = FastAPI(title="NeuroSleep AI Backend")

# --- MODEL LOADING ---
# The model utilizes a dual-stream architecture for EEG and EOG inputs
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
    """
    Standardizes and reshapes a list of epochs for the CNN-LSTM input.
    Expected input shape: (3000,) per epoch.
    Output shape: (Batch, 3000, 1).
    """
    processed = []
    for seg in data_list:
        seg = np.array(seg)
        # Scale to microvolts if signal is very low amplitude
        if np.max(np.abs(seg)) < 0.01: 
            seg *= 1e6
        
        # Z-score normalization
        if np.std(seg) > 0:
            seg = (seg - np.mean(seg)) / np.std(seg)
        
        processed.append(seg.reshape(3000, 1))
    
    return np.array(processed)

@app.post("/predict_batch")
async def predict_batch(data: InputData):
    """
    Receives JSON containing lists of EEG and EOG signals and returns AI predictions.
    """
    try:
        eeg_raw = data.eeg_data
        eog_raw = data.eog_data

        if not eeg_raw or not eog_raw:
            return {"error": "Missing input data"}

        # Prepare inputs for the functional API model
        X_eeg = preprocess_batch(eeg_raw)
        X_eog = preprocess_batch(eog_raw)

        # Run inference
        # Input keys must match those defined in your Keras model architecture
        raw_preds = model.predict(
            {'EEG_input': X_eeg, 'EOG_input': X_eog}, 
            verbose=0
        )

        # Extract class indices and confidence scores
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
