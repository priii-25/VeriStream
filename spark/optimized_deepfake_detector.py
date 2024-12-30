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
                # self.backbone = torch.jit.script(self.backbone)
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

    def predict_batch(self, frames, batch_size=8):
        all_scores = []
        face_batches = []
        current_batch = []
        
        def process_frame(frame):
            boxes, _ = self.face_detector.detect(frame)
            if boxes is None:
                return []
            faces = []
            for box in boxes.astype(int):
                face = frame[box[1]:box[3], box[0]:box[2]]
                if face.size != 0:
                    face = cv2.resize(face, (224, 224))
                    face = torch.from_numpy(face).permute(2, 0, 1).float() / 255.0
                    faces.append(face)
            return faces

        with concurrent.futures.ThreadPoolExecutor() as executor:
            processed_faces = list(executor.map(process_frame, frames))
            
        all_faces = [face for faces in processed_faces for face in faces]
        for i in range(0, len(all_faces), batch_size):
            batch = all_faces[i:i + batch_size]
            if batch:
                scores = self.process_batch(batch)
                all_scores.extend(scores)
                
        return np.mean(all_scores) if all_scores else 0.0, max(all_scores) if all_scores else 0.0