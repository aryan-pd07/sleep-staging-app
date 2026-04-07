# 🧠 NeuroSleep AI

> Clinical-grade automated sleep staging powered by deep learning.

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Visit%20App-blue?style=for-the-badge)](https://sleep-staging-neurosleep-frontend.onrender.com)
![Python](https://img.shields.io/badge/Python-3.10-green?style=for-the-badge&logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?style=for-the-badge&logo=tensorflow)
![Docker](https://img.shields.io/badge/Docker-Containerized-blue?style=for-the-badge&logo=docker)
![Tests](https://img.shields.io/badge/Tests-13%20Passed-brightgreen?style=for-the-badge&logo=pytest)

---

## 📌 Overview

NeuroSleep AI is a full-stack web application that automatically classifies sleep stages from raw EEG and EOG signals using a dual-stream CNN-LSTM deep learning model. It provides clinicians and researchers with an intuitive interface to upload polysomnography data, visualize sleep architecture, and generate detailed PDF reports — all in real time.

---

## ✨ Features

- 🔬 **AI Sleep Staging** — Classifies EEG + EOG signals into Wake, N1, N2, N3, and REM stages
- 📊 **Interactive Hypnogram** — Visualize sleep architecture across the full night
- 🆚 **Ground Truth Comparison** — Overlay AI predictions against doctor annotations
- 📄 **PDF Report Generation** — Download a clinical-style report with key metrics
- ☁️ **Cloud File Handling** — EDF files uploaded via Cloudinary, no server memory bottleneck
- 🔐 **Secure Authentication** — User login and signup via Supabase
- 🧪 **Unit Tested** — 13 tests covering preprocessing, API endpoints, and edge cases
- 🌐 **Fully Deployed** — Accessible from any device, anywhere

---

## 🏗️ Architecture

```
sleep-staging-app/
├── backend/               # FastAPI inference server
│   ├── api.py             # REST API endpoints
│   ├── test_backend.py    # Unit & integration tests (13 tests)
│   ├── model/             # CNN-LSTM Keras model
│   ├── Dockerfile
│   └── requirements.txt
├── frontend/              # Streamlit web app
│   ├── app.py             # Main application
│   ├── utils.py           # Helper functions
│   ├── assets/            # Logo and static files
│   ├── Dockerfile
│   └── requirements.txt
└── docker-compose.yml     # Local orchestration
```

### Model Architecture

The CNN-LSTM model (`best_cnn_lstm_model.keras`) uses a functional API with two parallel input streams:

| Input | Shape |
|-------|-------|
| EEG | (None, 3000, 1) |
| EOG | (None, 3000, 1) |

Both streams are processed independently through convolutional layers before being merged and passed through LSTM layers for temporal classification.

---

## 🚀 Live Demo

Try the app here: **[https://sleep-staging-neurosleep-frontend.onrender.com](https://sleep-staging-neurosleep-frontend.onrender.com)**

> ⏳ First load may take ~30 seconds (free tier spin-up)

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Model | TensorFlow / Keras (CNN-LSTM) |
| Backend | FastAPI + Uvicorn |
| Frontend | Streamlit |
| Auth | Supabase |
| File Storage | Cloudinary |
| Containerization | Docker |
| Deployment | Render |
| Testing | Pytest |

---

## 🧪 Testing

The backend includes **13 unit and integration tests** covering:

- Preprocessing pipeline (shape, normalization, edge cases)
- Class label and hypnogram mappings
- API endpoint behavior (valid data, missing data, correct predictions)

Run tests locally:

```bash
cd backend
pip install pytest httpx
pytest test_backend.py -v
```

Expected output:
```
13 passed in 0.47s
```

---

## 💻 Local Setup

### Prerequisites
- Docker & Docker Compose installed
- Supabase project credentials
- Cloudinary account credentials

### Run Locally

```bash
# Clone the repo
git clone https://github.com/aryan-pd07/sleep-staging-app.git
cd sleep-staging-app

# Set environment variables
export SUPABASE_URL=your_supabase_url
export SUPABASE_KEY=your_supabase_key
export CLOUDINARY_CLOUD_NAME=your_cloud_name
export CLOUDINARY_API_KEY=your_api_key
export CLOUDINARY_API_SECRET=your_api_secret

# Start both services
docker-compose up --build
```

Then open:
- Frontend → http://localhost:8501
- Backend API → http://localhost:8000

---

## 📡 API Reference

### `POST /predict_batch`

Runs inference on a batch of EEG/EOG epochs.

**Option A — Raw data (demo):**
```json
{
  "eeg_data": [[...], [...]],
  "eog_data": [[...], [...]]
}
```

**Option B — Cloud file URL (real EDF):**
```json
{
  "file_url": "https://res.cloudinary.com/...",
  "eeg_channel": "EEG Fpz-Cz",
  "eog_channel": "EOG horizontal"
}
```

**Response:**
```json
{
  "predictions": [2, 3, 4, 0, 1],
  "confidences": [91.2, 87.5, 95.1, 78.3, 88.9]
}
```

---

## ⚠️ Disclaimer

NeuroSleep AI is intended for **research support only** and does not constitute a medical diagnosis. Always consult a qualified clinician for medical decisions.

---

## 👨‍💻 Author

**Aryan Prasad**  
[GitHub](https://github.com/aryan-pd07) · [LinkedIn](https://linkedin.com/in/)

---

<p align="center">Made with ❤️ and way too little sleep 😴</p>