# # # model.py
# # import torch
# # from PIL import Image
# # import numpy as np
# # from facenet_pytorch import MTCNN
# # from emotiefflib.facial_analysis import EmotiEffLibRecognizer

# # # initialize model components only once
# # device = "cuda" if torch.cuda.is_available() else "cpu"
# # detector = MTCNN(keep_all=False, device=device, post_process=False)
# # recognizer = EmotiEffLibRecognizer(engine="torch", device=device)

# # def get_face_from_image(image: Image.Image):
# #     """Detect and crop the most confident face from the image."""
# #     img_array = np.array(image)
# #     boxes, probs = detector.detect(img_array)
# #     if boxes is None or probs is None or len(boxes) == 0:
# #         return None
# #     idx = int(np.argmax(probs))
# #     x1, y1, x2, y2 = boxes[idx].astype(int).tolist()
# #     if x2 <= x1 or y2 <= y1:
# #         return None
# #     face = img_array[y1:y2, x1:x2, :].copy()
# #     return Image.fromarray(face)

# # def predict_emotion(image_path: str):
# #     """Predict emotion from an image path."""
# #     try:
# #         image = Image.open(image_path).convert("RGB")
# #         face = get_face_from_image(image)
# #         if face is None:
# #             return None, None

# #         feats = recognizer.extract_features(np.array(face))
# #         labels, scores = recognizer.classify_emotions(feats, logits=False)
# #         label = labels[0]
# #         idx_map = recognizer.idx_to_emotion_class
# #         inv = {v: k for k, v in idx_map.items()}
# #         idx = inv[label]
# #         confidence = float(scores[0][idx])
# #         return label, confidence
# #     except Exception as e:
# #         print(f"Prediction error: {e}")
# #         return None, None






# # model.py
# import os
# import numpy as np
# import pandas as pd
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import Dataset, DataLoader
# from torchvision.models import resnet18, ResNet18_Weights
# from torchvision import models, transforms
# from PIL import Image
# from facenet_pytorch import MTCNN

# # Paths and settings
# FER_CSV = "fer2013.csv"  # you must download this and place in project root
# MODEL_SAVE_PATH = "emotion_model.pth"
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# # Emotion labels (matching FER2013 integers 0-6)
# EMOTION_MAP = {
#     0: "angry",
#     1: "disgust",
#     2: "fear",
#     3: "happy",
#     4: "sad",
#     5: "surprise",
#     6: "neutral"
# }

# # Face detector
# mtcnn = MTCNN(keep_all=False, device=DEVICE, post_process=False)

# # DataSet wrapper
# class FERDataset(Dataset):
#     def __init__(self, df, transform=None):
#         self.df = df
#         self.transform = transform

#     def __len__(self):
#         return len(self.df)

#     def __getitem__(self, idx):
#         row = self.df.iloc[idx]
#         pixels = np.fromstring(row["pixels"], dtype=int, sep=" ")
#         image = Image.fromarray(pixels.reshape(48, 48).astype(np.uint8), mode="L").convert("RGB")
#         if self.transform:
#             image = self.transform(image)
#         label = row["emotion"]
#         return image, label

# def build_model(num_classes=7):
#     model = resnet18(weights=ResNet18_Weights.DEFAULT)
#     for param in model.parameters():
#         param.requires_grad = False
#     model.fc = nn.Sequential(
#         nn.Linear(model.fc.in_features, 256),
#         nn.ReLU(),
#         nn.Dropout(0.3),
#         nn.Linear(256, num_classes)
#     )
#     return model.to(DEVICE)

# def train_model(epochs=5, batch_size=64, lr=1e-3):
#     df = pd.read_csv(FER_CSV)
#     # Using only Training rows (optional: filter Usage column)
#     train_df = df[df["Usage"] == "Training"]
#     val_df = df[df["Usage"] == "PublicTest"]

#     transform = transforms.Compose([
#         transforms.Resize((224, 224)),
#         transforms.ToTensor(),
#         transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
#     ])

#     train_ds = FERDataset(train_df, transform=transform)
#     val_ds = FERDataset(val_df, transform=transform)
#     train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
#     val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

#     model = build_model()
#     criterion = nn.CrossEntropyLoss()
#     optimizer = optim.Adam(model.fc.parameters(), lr=lr)

#     for epoch in range(epochs):
#         model.train()
#         running_loss = 0.0
#         for images, labels in train_loader:
#             images = images.to(DEVICE)
#             labels = labels.to(DEVICE)
#             optimizer.zero_grad()
#             outputs = model(images)
#             loss = criterion(outputs, labels)
#             loss.backward()
#             optimizer.step()
#             running_loss += loss.item()
#         print(f"Epoch {epoch+1}/{epochs} train_loss={running_loss/len(train_loader):.4f}")

#         # validation
#         model.eval()
#         correct = 0
#         total = 0
#         with torch.no_grad():
#             for images, labels in val_loader:
#                 images = images.to(DEVICE)
#                 labels = labels.to(DEVICE)
#                 outputs = model(images)
#                 _, preds = torch.max(outputs, 1)
#                 total += labels.size(0)
#                 correct += (preds == labels).sum().item()
#         print(f"Validation accuracy: {correct/total:.4f}")

#     # Save model state
#     torch.save(model.state_dict(), MODEL_SAVE_PATH)
#     print(f"Model saved to {MODEL_SAVE_PATH}")
#     return model

# # Prediction interface (to use in app.py)
# def predict_emotion(image_path: str):
#     # detect face
#     image = Image.open(image_path).convert("RGB")
#     face_img = mtcnn(image)
#     if face_img is None:
#         return None, 0.0
#     # preprocess face
#     transform = transforms.Compose([
#         transforms.Resize((224,224)),
#         transforms.ToTensor(),
#         transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
#     ])
#     face = transform(face_img).unsqueeze(0).to(DEVICE)

#     model = build_model()
#     model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=DEVICE))
#     model.eval()
#     with torch.no_grad():
#         outputs = model(face)
#         _, pred = torch.max(outputs, 1)
#         label = EMOTION_MAP[int(pred.item())]
#         score = float(torch.softmax(outputs, dim=1)[0][pred].item())
#     return label, score

# # if run as script, train and save model
# if __name__ == "__main__":
#     train_model(epochs=3)  # you can increase epochs for better accuracy






# model.py
"""
Inference utilities using facenet-pytorch (MTCNN) + emotiefflib (pretrained).
Provides:
 - initialize_recognizer(device, model_name)
 - predict_emotion_from_image(image_path) -> (label:str, confidence:float)
 - optionally saves a small metadata file model_info.pkl for submission
"""

import os
import joblib
import numpy as np
from PIL import Image
from typing import Tuple, Optional

# prefer GPU if available (will fall back to cpu)
import torch
from facenet_pytorch import MTCNN
from emotiefflib.facial_analysis import EmotiEffLibRecognizer, get_model_list

MODEL_INFO_PATH = "model.pkl"

# choose device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Initialize MTCNN once (keep_all=False -> single face)
_mtcnn = MTCNN(keep_all=False, post_process=False, min_face_size=40, device=DEVICE)

# Choose a good emotiefflib model name (take first available)
_AVAILABLE_MODELS = get_model_list()
if not _AVAILABLE_MODELS:
    raise RuntimeError("No models returned by emotiefflib.get_model_list()")
_DEFAULT_MODEL_NAME = _AVAILABLE_MODELS[0]

# Initialize recognizer once
_RECOGNIZER = EmotiEffLibRecognizer(engine="torch", model_name=_DEFAULT_MODEL_NAME, device=DEVICE)

if not os.path.exists(MODEL_INFO_PATH):
    info = {
        "backend": "emotiefflib",
        "model_name": _DEFAULT_MODEL_NAME,
        "face_detector": "facenet-pytorch MTCNN",
        "device": DEVICE
    }
    joblib.dump(info, MODEL_INFO_PATH)


def _crop_first_face(pil_img: Image.Image) -> Optional[np.ndarray]:
    """
    Use MTCNN to detect the best face and return a (H,W,3) uint8 numpy array crop.
    Returns None if no face detected.
    """
    img_arr = np.asarray(pil_img)
    boxes, probs = _mtcnn.detect(img_arr, landmarks=False)
    if boxes is None or probs is None:
        return None
    idx = int(np.argmax(probs))
    x1, y1, x2, y2 = boxes[idx].astype(int).tolist()
    # clamp coords
    h, w = img_arr.shape[:2]
    x1 = max(0, x1); y1 = max(0, y1); x2 = min(w, x2); y2 = min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return None
    face = img_arr[y1:y2, x1:x2, :].copy()
    return face


def predict_emotion(image_path: str) -> Tuple[Optional[str], float]:
    """
    Predict the emotion from an image file path.
    Returns (label, confidence) where label is None if no face detected.
    Confidence is in [0,1].
    """
    try:
        pil = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"[model] Unable to open image {image_path}: {e}")
        return None, 0.0

    face = _crop_first_face(pil)
    if face is None:
        # no face found
        return None, 0.0

    # emotiefflib convenience: recognizer.extract_features accepts numpy arrays / PIL
    # Provide face (H,W,3) numpy array
    try:
        feats = _RECOGNIZER.extract_features(face)
        labels, scores = _RECOGNIZER.classify_emotions(feats, logits=False)
        # labels is a list of label strings; scores is list of arrays (probabilities per class)
        predicted_label = labels[0]
        # Map label string to index using recognizer mapping
        idx_map = _RECOGNIZER.idx_to_emotion_class  # dict like { "happy": 0, ... }? check lib
        # The library in earlier code used mapping k->v; invert to get index by label
        inv_map = {v: k for k, v in idx_map.items()}
        if predicted_label not in inv_map:
            # If mapping shape different, find best confidence by searching label name in labels list
            # fallback: take max score index
            probs = scores[0]
            best_idx = int(np.argmax(probs))
            # get class name by mapping: find key whose value == best_idx
            label_name = None
            for k, v in idx_map.items():
                if v == best_idx:
                    label_name = k
                    break
            if label_name is None:
                # give fallback
                return predicted_label, float(np.max(scores[0]))
            confidence = float(scores[0][best_idx])
            return label_name, confidence

        idx = inv_map[predicted_label]
        confidence = float(scores[0][idx]) * 100
        return predicted_label, round(confidence, 2)

    except Exception as e:
        print(f"[model] emotiefflib prediction error: {e}")
        return None, 0.0
