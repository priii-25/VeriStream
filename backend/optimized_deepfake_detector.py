# backend/optimized_deepfake_detector.py
import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet
import cv2
import numpy as np
from facenet_pytorch import MTCNN
import torch.nn.functional as F
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('deepfake_detector')

class OptimizedDeepfakeDetector(nn.Module):
    def __init__(self):
        """Initialize the deepfake detector with EfficientNet and MTCNN."""
        super().__init__()
        try:
            self.backbone = EfficientNet.from_pretrained('efficientnet-b4')
            self.face_detector = MTCNN(
                keep_all=True,
                min_face_size=60,
                thresholds=[0.6, 0.7, 0.7],
                device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            )
            self.classifier = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(1792, 2)
            )
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.to(self.device)
            self.eval()
            self.batch_size = 16

            if torch.cuda.is_available():
                self.backbone = self.backbone.half()
                self.classifier = self.classifier.half()
            logger.info("Deepfake detector initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize deepfake detector: {str(e)}")
            raise

    @torch.no_grad()
    def process_batch(self, face_batch):
        """Process a batch of face images."""
        if not face_batch:
            return []
        try:
            batch_tensor = torch.stack(face_batch).to(self.device)
            if torch.cuda.is_available():
                batch_tensor = batch_tensor.half()

            features = self.backbone.extract_features(batch_tensor)
            outputs = self.classifier(features)
            scores = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
            return scores.tolist()
        except Exception as e:
            logger.error(f"Error processing batch: {str(e)}")
            return []

    def predict_batch(self, frames):
        """Predict deepfake scores for a batch of frames."""
        per_frame_scores = []
        all_faces = []
        frame_indices = []

        try:
            for frame_idx, frame in enumerate(frames):
                boxes, _ = self.face_detector.detect(frame)
                if boxes is not None:
                    for box in boxes.astype(int):
                        x1, y1, x2, y2 = box
                        x1, y1 = max(0, x1), max(0, y1)
                        x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)

                        if x1 < x2 and y1 < y2:
                            face = frame[y1:y2, x1:x2]
                            if face.size != 0:
                                face = cv2.resize(face, (224, 224))
                                face = torch.from_numpy(face).permute(2, 0, 1).float() / 255.0
                                all_faces.append(face)
                                frame_indices.append(frame_idx)

            per_face_scores = []
            for i in range(0, len(all_faces), self.batch_size):
                batch = all_faces[i:i + self.batch_size]
                scores = self.process_batch(batch)
                per_face_scores.extend(scores)

            per_frame_scores = [0.0] * len(frames)
            for idx, face_score in enumerate(per_face_scores):
                frame_idx = frame_indices[idx]
                if face_score > per_frame_scores[frame_idx]:
                    per_frame_scores[frame_idx] = face_score

            return per_frame_scores
        except Exception as e:
            logger.error(f"Error predicting batch: {str(e)}")
            return [0.0] * len(frames)

# For testing standalone
if __name__ == "__main__":
    detector = OptimizedDeepfakeDetector()
    frame = cv2.imread("test_frame.jpg")
    scores = detector.predict_batch([frame])
    print(f"Deepfake scores: {scores}")