import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet
import cv2
import numpy as np
from facenet_pytorch import MTCNN
import torch.nn.functional as F  # Import the functional module
from torch.quantization import quantize_dynamic
from torch.jit import script
import concurrent.futures #This wasn't used

class OptimizedDeepfakeDetector(nn.Module):
    def __init__(self, batch_size=16):  # Add batch_size as an argument to __init__
        super().__init__()
        self.backbone = EfficientNet.from_pretrained('efficientnet-b4')
        self.face_detector = MTCNN(keep_all=True,
                                 min_face_size=60,
                                 thresholds=[0.6, 0.7, 0.7],
                                 device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(1792, 2)
        )

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)
        self.eval()
        self.batch_size = batch_size  # Store batch_size as an attribute

        if torch.cuda.is_available():
            self.backbone = self.backbone.half()
            self.classifier = self.classifier.half()

        if not torch.cuda.is_available():
            try:
                print("JIT compiling backbone...")
                self.backbone = torch.jit.script(self.backbone) # Added
            except Exception as e:
                print(f"Warning: JIT compilation failed: {e}")

    @torch.no_grad()
    def process_batch(self, face_batch):
        if not face_batch:
            return []
        batch_tensor = torch.stack(face_batch).to(self.device)
        if torch.cuda.is_available():
            batch_tensor = batch_tensor.half()

        features = self.backbone.extract_features(batch_tensor)
        outputs = self.classifier(features)
        scores = F.softmax(outputs, dim=1)[:, 1].cpu().numpy() #use torch.nn.functional
        return scores.tolist()

    def predict_batch(self, frames, batch_size=16):  # Add batch_size argument, with a default
        per_frame_scores = []
        all_faces = []
        frame_indices = []

        for frame_idx, frame in enumerate(frames):
            boxes, _ = self.face_detector.detect(frame)
            if boxes is not None:
                for box in boxes: #removed astype(int)
                    if box is not None:
                      # Ensure box coordinates are within frame boundaries
                      box = [int(b) for b in box] # NOW we convert
                      height, width = frame.shape[:2]
                      x1 = max(0, box[0])
                      y1 = max(0, box[1])
                      x2 = min(width, box[2])
                      y2 = min(height, box[3])
                      # Ensure the bounding box has valid dimensions
                      if x1 < x2 and y1 < y2:
                        face = frame[y1:y2, x1:x2]
                        if face.size != 0:
                            face = cv2.resize(face, (224, 224))
                            face = torch.from_numpy(face).permute(2, 0, 1).float() / 255.0
                            all_faces.append(face)
                            frame_indices.append(frame_idx)

        # Process all faces in batches
        per_face_scores = []
        for i in range(0, len(all_faces), batch_size):  # Use the passed-in batch_size
           batch = all_faces[i:i + batch_size]
           scores = self.process_batch(batch)
           per_face_scores.extend(scores)

        # Aggregate max score per frame
        per_frame_scores = [0.0] * len(frames)
        for idx, face_score in enumerate(per_face_scores):
            frame_idx = frame_indices[idx]
            if face_score > per_frame_scores[frame_idx]:
                per_frame_scores[frame_idx] = face_score

        return per_frame_scores