# # model.py — utilities for detection, evaluation, and a minimal training scaffold
# import argparse
# import os
# from typing import List
# import numpy as np
# from PIL import Image

# from facenet_pytorch import MTCNN
# from emotiefflib.facial_analysis import EmotiEffLibRecognizer, get_model_list

# # Optional imports for training scaffold
# try:
#     import torch
#     import torch.nn as nn
#     import torch.optim as optim
#     from torch.utils.data import Dataset, DataLoader
#     TORCH_AVAILABLE = True
# except Exception:
#     TORCH_AVAILABLE = False


# def locate_faces(img_np: np.ndarray, device: str = "cpu") -> List[np.ndarray]:
#     """Return a list of cropped faces from the input RGB image (np.ndarray)."""
#     detector = MTCNN(keep_all=True, device=device)
#     boxes, _ = detector.detect(img_np)
#     faces = []
#     if boxes is None:
#         return faces
#     for box in boxes:
#         x1, y1, x2, y2 = [int(v) for v in box[:4]]
#         if x2 > x1 and y2 > y1:
#             faces.append(img_np[y1:y2, x1:x2, :].copy())
#     return faces


# def infer_file(image_path: str, model_name: str = None, device: str = "cpu") -> None:
#     if model_name is None:
#         model_name = get_model_list()[0]

#     if not os.path.isfile(image_path):
#         raise FileNotFoundError(f"Image not found: {image_path}")

#     pil = Image.open(image_path).convert("RGB")
#     img_np = np.array(pil)
#     faces = locate_faces(img_np, device=device)
#     if not faces:
#         print("No faces were detected.")
#         return

#     recognizer = EmotiEffLibRecognizer(engine="torch", model_name=model_name, device=device)
#     # batch-friendly: recognizer.extract_features accepts list of faces
#     feats = recognizer.extract_features(faces)
#     labels, scores = recognizer.classify_emotions(feats, logits=False)

#     for i, lbl in enumerate(labels):
#         idx_map = recognizer.idx_to_emotion_class
#         inv = {v: k for k, v in idx_map.items()}
#         idx = inv[lbl]
#         conf = float(scores[i][idx])
#         print(f"face={i} label={lbl} confidence={conf:.4f}")


# # Minimal dataset and training scaffold (illustrative)
# class SimpleFaceDataset(Dataset):
#     """A tiny dataset wrapper — expects a list of (face_np, label_idx) tuples."""
#     def __init__(self, examples):
#         self.examples = examples

#     def __len__(self):
#         return len(self.examples)

#     def __getitem__(self, i):
#         face_np, lbl = self.examples[i]
#         # convert to CHW float tensor normalized to [0,1]
#         tensor = torch.from_numpy(face_np.transpose(2, 0, 1)).float() / 255.0
#         return tensor, torch.tensor(lbl, dtype=torch.long)


# def training_demo(save_path: str = "trained_head.pt", epochs: int = 5, device: str = "cpu"):
#     """A tiny demo showing how one might train a small classifier on top of extracted features.

#     This is illustrative and not intended as a full training pipeline.
#     """
#     if not TORCH_AVAILABLE:
#         raise RuntimeError("PyTorch not available; cannot run training demo.")

#     print("=== training demo (toy) ===")
#     # Normally you'd build a dataset from real labeled faces.
#     # Here we create a random toy dataset to illustrate the flow.
#     num_classes = 7
#     examples = []
#     for _ in range(40):
#         face = np.random.randint(0, 255, size=(64, 64, 3), dtype=np.uint8)
#         lbl = np.random.randint(0, num_classes)
#         examples.append((face, lbl))

#     ds = SimpleFaceDataset(examples)
#     loader = DataLoader(ds, batch_size=8, shuffle=True)

#     # Simple model head: flatten -> linear -> softmax
#     class Head(nn.Module):
#         def __init__(self, out_dim):
#             super().__init__()
#             self.net = nn.Sequential(
#                 nn.Flatten(),
#                 nn.Linear(64 * 64 * 3, 256),
#                 nn.ReLU(),
#                 nn.Linear(256, out_dim)
#             )

#         def forward(self, x):
#             return self.net(x)

#     model = Head(num_classes).to(device)
#     opt = optim.Adam(model.parameters(), lr=1e-3)
#     loss_fn = nn.CrossEntropyLoss()

#     for ep in range(epochs):
#         running = 0.0
#         for xb, yb in loader:
#             xb = xb.to(device)
#             yb = yb.to(device)
#             logits = model(xb)
#             loss = loss_fn(logits, yb)
#             opt.zero_grad()
#             loss.backward()
#             opt.step()
#             running += loss.item()
#         print(f"Epoch {ep+1}/{epochs} loss={running/len(loader):.4f}")

#     torch.save(model.state_dict(), save_path)
#     print(f"Saved demo head to {save_path}")


# def build_cli():
#     p = argparse.ArgumentParser(prog="emotion-tools")
#     p.add_argument("--image", help="Path to an image for inference")
#     p.add_argument("--model", help="EmotiEff model name (optional)")
#     p.add_argument("--device", default="cpu", help="cpu or cuda")
#     p.add_argument("--train-demo", action="store_true", help="Run a toy training demo and save a small head")
#     p.add_argument("--save", default="trained_head.pt", help="Output path for demo training")
#     return p


# def main():
#     args = build_cli().parse_args()
#     if args.train_demo:
#         training_demo(save_path=args.save, device=args.device)
#         return

#     if args.image:
#         infer_file(args.image, model_name=args.model, device=args.device)
#     else:
#         print("No action specified. Use --image <path> or --train-demo to run the toy trainer.")


# if __name__ == "__main__":
#     main()



# model.py
import torch
from PIL import Image
import numpy as np
from facenet_pytorch import MTCNN
from emotiefflib.facial_analysis import EmotiEffLibRecognizer

# initialize model components only once
device = "cuda" if torch.cuda.is_available() else "cpu"
detector = MTCNN(keep_all=False, device=device, post_process=False)
recognizer = EmotiEffLibRecognizer(engine="torch", device=device)

def get_face_from_image(image: Image.Image):
    """Detect and crop the most confident face from the image."""
    img_array = np.array(image)
    boxes, probs = detector.detect(img_array)
    if boxes is None or probs is None or len(boxes) == 0:
        return None
    idx = int(np.argmax(probs))
    x1, y1, x2, y2 = boxes[idx].astype(int).tolist()
    if x2 <= x1 or y2 <= y1:
        return None
    face = img_array[y1:y2, x1:x2, :].copy()
    return Image.fromarray(face)

def predict_emotion(image_path: str):
    """Predict emotion from an image path."""
    try:
        image = Image.open(image_path).convert("RGB")
        face = get_face_from_image(image)
        if face is None:
            return None, None

        feats = recognizer.extract_features(np.array(face))
        labels, scores = recognizer.classify_emotions(feats, logits=False)
        label = labels[0]
        idx_map = recognizer.idx_to_emotion_class
        inv = {v: k for k, v in idx_map.items()}
        idx = inv[label]
        confidence = float(scores[0][idx])
        return label, confidence
    except Exception as e:
        print(f"Prediction error: {e}")
        return None, None
