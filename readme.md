# Emotion Detection Web Application 🎭

This project detects human emotions from facial images using **EmotiEffLib** and **facenet-pytorch**.

## 📦 Project Description

Users can upload an image or use their webcam to detect facial emotions such as _happy, sad, angry, neutral_, etc.  
Predictions are stored in a local SQLite database with names, timestamps, confidence scores, and uploaded images.

---

## 🧠 Features

- Face detection using **MTCNN**
- Emotion classification via **EmotiEffLibRecognizer**
- Streamlit UI (image upload or live camera)
- Minimal toy training example in `model.py`

---

## ⚙️ Setup Instructions

1. Clone this repository:
   ```bash
   git clone https://github.com/Beloved-adetola/csc-331-ai-emotion-detector.git
   cd <folder_name>
    pip install -r requirements.txt
    streamlit run app.py
   ```
