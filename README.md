# Sleep Stage Classification App

A Streamlit application serving a dual-stream CNN-LSTM model for automatic sleep staging.

## Model Architecture
The app loads `best_cnn_lstm_model.keras`, which utilizes a functional API with two inputs:
- **EEG Input:** (None, 3000, 1)
- **EOG Input:** (None, 3000, 1)

## Setup Instructions

1. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt