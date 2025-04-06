#backend/optimized_deepfake_detector.py
import torch
import torch.nn as nn
from transformers import Dinov2Model
from torchvision import transforms
import cv2
import numpy as np
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('deepfake_detector')

class OptimizedDeepfakeDetector(nn.Module):
    def __init__(self, model_path="/Users/kartik/Desktop/vs/VeriStream/backend/dinov2_deepfake_final.pt"):
        """Initialize the fine-tuned DINOv2 model for deepfake detection on CPU."""
        super().__init__()
        try:
            self.dinov2 = Dinov2Model.from_pretrained("facebook/dinov2-small")
            self.classifier = nn.Sequential(
                nn.Linear(384, 128),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(128, 2)
            )
            self.device = torch.device("cpu")  # Explicitly set to CPU
            # Load fine-tuned weights
            state_dict = torch.load(model_path, map_location=self.device)
            self.load_state_dict(state_dict)
            self.to(self.device)
            self.eval()
            logger.info(f"DINOv2 Deepfake Detector initialized on {self.device}")
        except Exception as e:
            logger.error(f"Failed to initialize deepfake detector: {str(e)}")
            raise

    def forward(self, x):
        """Define the forward pass for the model."""
        features = self.dinov2(x).last_hidden_state[:, 0]  # Extract CLS token
        logits = self.classifier(features)
        return logits

    @torch.no_grad()
    def process_batch(self, face_batch):
        """Process a batch of face images on CPU."""
        if not face_batch:
            return []
        try:
            batch_tensor = torch.stack(face_batch).to(self.device)  # Already CPU
            outputs = self(batch_tensor)
            scores = torch.softmax(outputs, dim=1)[:, 1].numpy()  # No .cpu() needed
            return scores.tolist()
        except Exception as e:
            logger.error(f"Error processing batch: {str(e)}")
            return []

    @torch.no_grad()
    def predict_batch(self, frames):
        """Predict deepfake scores for a batch of frames on CPU."""
        try:
            faces = []
            for frame in frames:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (224, 224))
                frame = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
                frame = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(frame)
                faces.append(frame)
            return self.process_batch(faces)
        except Exception as e:
            logger.error(f"Error predicting batch: {str(e)}")
            return [0.0] * len(frames)

# Test standalone
if __name__ == "__main__":
    detector = OptimizedDeepfakeDetector("/Users/kartik/Desktop/vs/VeriStream/backend/dinov2_deepfake_final.pt")
    frame = cv2.imread("/Users/kartik/Desktop/vs/VeriStream/backend/test.jpeg")
    if frame is not None:
        scores = detector.predict_batch([frame])
        print(f"Deepfake scores: {scores}")
    else:
        print("Failed to load test frame")