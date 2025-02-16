#optimized_deepfake_detector.py
import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet
import cv2
import numpy as np
from facenet_pytorch import MTCNN
import torch.nn.functional as F
from torch.quantization import quantize_dynamic
from torch.jit import script
import concurrent.futures

class OptimizedDeepfakeDetector(nn.Module):
    def __init__(self):
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
        
        if torch.cuda.is_available():
            self.backbone = self.backbone.half()
            self.classifier = self.classifier.half()
        
        if not torch.cuda.is_available():
            try:
                print("JIT compiling backbone...")
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
        scores = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
        return scores.tolist()

    # In optimized_deepfake_detector.py
    def predict_batch(self, frames):
        per_frame_scores = []
        all_faces = []
        frame_indices = []

        for frame_idx, frame in enumerate(frames):
            boxes, _ = self.face_detector.detect(frame)
            if boxes is not None:
                faces = []
                for box in boxes.astype(int):
                    face = frame[box[1]:box[3], box[0]:box[2]]
                    if face.size != 0:
                        face = cv2.resize(face, (224, 224))
                        face = torch.from_numpy(face).permute(2, 0, 1).float() / 255.0
                        all_faces.append(face)
                        frame_indices.append(frame_idx)

    # Process all faces in batches
        per_face_scores = []
        for i in range(0, len(all_faces), self.batch_size):
            batch = all_faces[i:i + self.batch_size]
            scores = self.process_batch(batch)
            per_face_scores.extend(scores)

    # Aggregate max score per frame
        per_frame_scores = [0.0] * len(frames)
        for idx, face_score in enumerate(per_face_scores):
            frame_idx = frame_indices[idx]
            if face_score > per_frame_scores[frame_idx]:
                per_frame_scores[frame_idx] = face_score

        return per_frame_scores    
    