import os
import cv2
import torch
import numpy as np
from facenet_pytorch import MTCNN, InceptionResnetV1
import logging

logger = logging.getLogger(__name__)

class FaceVerifier:
    """
    Enhanced face verifier using facenet-pytorch (MTCNN + InceptionResnetV1).
    """
    def __init__(self, database_dir: str, threshold: float = 1.0, num_frames: int = 5):
        self.database_dir = database_dir
        self.threshold = threshold
        self.num_frames = num_frames
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.mtcnn = MTCNN(keep_all=True, device=self.device)
        self.resnet = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)
        self.known_embeddings = []
        self.names = []
        os.makedirs(self.database_dir, exist_ok=True)
        self._load_database()
        if self.known_embeddings:
            self.known_embeddings = np.vstack(self.known_embeddings)
        else:
            # no known faces
            self.known_embeddings = np.zeros((0, 512))

    def _load_database(self):
        for filename in os.listdir(self.database_dir):
            filepath = os.path.join(self.database_dir, filename)
            if not os.path.isfile(filepath):
                continue
            try:
                img = cv2.imread(filepath)
                if img is None:
                    logger.warning(f"Cannot read image {filepath}")
                    continue
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                face_tensors = self.mtcnn(img_rgb)
                if face_tensors is None:
                    continue
                # select first face
                if isinstance(face_tensors, torch.Tensor):
                    face_tensor = face_tensors[0]
                else:
                    face_tensor = face_tensors
                with torch.no_grad():
                    emb = self.resnet(face_tensor.unsqueeze(0).to(self.device)).squeeze(0).cpu().numpy()
                self.known_embeddings.append(emb)
                name, _ = os.path.splitext(filename)
                self.names.append(name)
            except Exception as e:
                logger.warning(f"Failed processing {filepath}: {e}")

    def verify_video(self, video_path: str) -> dict:
        video_capture = cv2.VideoCapture(video_path)
        frame_count = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if frame_count == 0:
            video_capture.release()
            return {"verified": False, "recognized_names": []}
        indices = np.linspace(0, frame_count - 1, self.num_frames, dtype=int)
        recognized = set()
        for idx in indices:
            video_capture.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = video_capture.read()
            if not ret:
                continue
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_tensors = self.mtcnn(img_rgb)
            if face_tensors is None:
                continue
            tensors = face_tensors if isinstance(face_tensors, torch.Tensor) else [face_tensors]
            for face_tensor in tensors:
                with torch.no_grad():
                    emb = self.resnet(face_tensor.unsqueeze(0).to(self.device)).squeeze(0).cpu().numpy()
                if self.known_embeddings.size == 0:
                    continue
                dists = np.linalg.norm(self.known_embeddings - emb, axis=1)
                best_idx = int(np.argmin(dists))
                if dists[best_idx] <= self.threshold:
                    recognized.add(self.names[best_idx])
        video_capture.release()
        return {"verified": bool(recognized), "recognized_names": list(recognized)}
