# # Streamlit frontend/backend for Emotion Detection
# from __future__ import annotations
# import sqlite3
# import io
# from datetime import datetime, UTC
# from typing import Optional, Tuple, List

# import numpy as np
# from PIL import Image
# import streamlit as st
# from facenet_pytorch import MTCNN
# from emotiefflib.facial_analysis import EmotiEffLibRecognizer, get_model_list

# DATABASE_FILE = "face_emotion_app.db"

# def open_or_create_db(path: str = DATABASE_FILE) -> sqlite3.Connection:
#     """Ensure database exists and create the table if necessary."""
#     conn = sqlite3.connect(path, check_same_thread=False)
#     c = conn.cursor()
#     c.execute(
#         """
#         CREATE TABLE IF NOT EXISTS emotions (
#             id INTEGER PRIMARY KEY,
#             username TEXT NOT NULL,
#             ts TEXT NOT NULL,
#             picture BLOB NOT NULL,
#             predicted TEXT NOT NULL,
#             score REAL NOT NULL
#         )
#         """
#     )
#     conn.commit()
#     return conn

# def add_emotion(conn: sqlite3.Connection, username: str, img_bytes: bytes, mood: str, conf: float) -> None:
#     cur = conn.cursor()
#     cur.execute(
#         "INSERT INTO emotions (username, ts, picture, predicted, score) VALUES (?, ?, ?, ?, ?)",
#         (username, datetime.now(UTC).isoformat(), img_bytes, mood, conf),
#     )
#     conn.commit()


# # def get_latest_records(conn: sqlite3.Connection, limit: int = 50) -> List[tuple]:
# #     cur = conn.cursor()
# #     rows = cur.execute("SELECT username, ts, predicted, score FROM records ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
# #     return rows

# # Face Detection
# def get_face_from_image(img_arr: np.ndarray, device: str = "cpu") -> Optional[np.ndarray]:
#     # Use MTCNN to find the most confident face and return cropped face array
#     detector = MTCNN(keep_all=False, device=device, post_process=False, min_face_size=40)
#     boxes, probs = detector.detect(img_arr)
#     if boxes is None or probs is None:
#         return None
#     # pick highest prob
#     idx = int(np.argmax(probs))
#     x1, y1, x2, y2 = boxes[idx].astype(int).tolist()
#     if x2 <= x1 or y2 <= y1:
#         return None
#     face = img_arr[y1:y2, x1:x2, :].copy()
#     return face

# # Face Prediction
# def predict_with_model(model: EmotiEffLibRecognizer, face_img: np.ndarray) -> Tuple[str, float]:
#     feats = model.extract_features(face_img)
#     labels, scores = model.classify_emotions(feats, logits=False)
#     label = labels[0]
#     # map label name to index using recognizer's mapping
#     mapping = model.idx_to_emotion_class
#     reverse = {v: k for k, v in mapping.items()}
#     idx = reverse[label]
#     confidence = float(scores[0][idx])
#     return label, confidence


# def render_ui() -> None:
#     st.set_page_config(page_title="Face Emotion Detector — Stream", page_icon="🧠", layout="centered")

#     # Custom Styling
#     st.markdown("""
#         <style>
#         .stApp {
#             background-color: #11113f;
#             color: #EAEAEA;
#         }
#         .stButton>button {
#             background-color: #00ADB5;
#             color: white;
#             border-radius: 8px;
#             font-weight: bold;
#         }
#         </style>
#     """, unsafe_allow_html=True)

#     st.header("Face Emotion Detector")
#     st.caption("Detect human emotions from facial expressions in images")

#     conn = open_or_create_db()

#     # Model Initialization
#     device = "cpu"
#     model_choice = get_model_list()[0]  # pick first model
#     recognizer = EmotiEffLibRecognizer(engine="torch", model_name=model_choice, device=device)

#     # User input
#     name = st.text_input("Your name: ", max_chars=30, placeholder="John Maxwell")

#     img_arr: Optional[np.ndarray] = None
#     raw_blob: Optional[bytes] = None

#     uploaded = st.file_uploader("Choose Image", type=["jpg", "jpeg", "png"])

#     if uploaded:
#         raw_blob = uploaded.getvalue()
#         st.image(uploaded, caption="Uploaded", use_container_width=True)
#         pil = Image.open(io.BytesIO(raw_blob)).convert("RGB")
#         img_arr = np.array(pil)

#     if img_arr is not None and raw_blob is not None:
#         face = get_face_from_image(img_arr, device=device)
#         if face is None:
#             st.warning("No face found or try a clearer image")
#         else:
#             label, score = predict_with_model(recognizer, face)
#             st.success(f"Detected: **{label}** — confidence: **{score:.3f}**")
#             col1, col2 = st.columns(2)
#             with col1:
#                 st.image(Image.fromarray(face), caption="Cropped face", use_container_width=True)
#             with col2:
#                 if st.button("Save to DB"):
#                     if not name:
#                         st.error("Name is required before saving.")
#                     else:
#                         add_emotion(conn, name, raw_blob, label, score)
#                         st.info("Saved successfully to database")

# if __name__ == "__main__":
#     render_ui()


# app.py
import os
import sqlite3
from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for, flash
from model import predict_emotion

app = Flask(__name__)
app.secret_key = "supersecretkey"

# Folder for uploaded images
UPLOAD_FOLDER = os.path.join("static", "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Database file
DB_FILE = "face_emotion_app.db"

# --- Initialize database ---
def init_db():
    with sqlite3.connect(DB_FILE) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL,
                ts TEXT NOT NULL,
                predicted TEXT NOT NULL,
                score REAL NOT NULL,
                picture_path TEXT NOT NULL
            )
        """)
        conn.commit()

init_db()

# --- Save record helper ---
def save_record(username, predicted, score, picture_filename):
    picture_path = f"static/uploads/{picture_filename}"  # stored path
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with sqlite3.connect(DB_FILE) as conn:
        conn.execute("""
            INSERT INTO records (username, ts, predicted, score, picture_path)
            VALUES (?, ?, ?, ?, ?)
        """, (username, ts, predicted, score, picture_path))
        conn.commit()

# --- Get recent records ---
def get_recent_records(limit=10):
    with sqlite3.connect(DB_FILE) as conn:
        cur = conn.cursor()
        cur.execute(
            "SELECT username, ts, predicted, score, picture_path FROM records ORDER BY id DESC LIMIT ?",
            (limit,)
        )
        return cur.fetchall()

# --- Routes ---
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        username = request.form.get("username")
        image = request.files.get("image")

        if not username or not image:
            flash("Please enter your name and upload an image.")
            return redirect(url_for("index"))

        # Save uploaded image
        filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{image.filename}"
        save_path = os.path.join(UPLOAD_FOLDER, filename).replace("\\", "/")
        image.save(save_path)

        # Predict emotion
        label, score = predict_emotion(save_path)
        if label is None:
            flash("No face detected. Try another image.")
            return redirect(url_for("index"))

        # Use save_record() to store in DB
        save_record(username, label, score, filename)

        flash(f"Detected emotion: {label} (confidence: {score:.2f}%)")
        return redirect(url_for("index"))

    # Display recent detections
    recent = get_recent_records()
    return render_template("index.html", recent=recent)

if __name__ == "__main__":
    app.run(debug=True)
