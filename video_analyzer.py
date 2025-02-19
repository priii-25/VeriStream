import cv2
import time
import numpy as np
import logging
from typing import Tuple, List, Dict
from models import load_models
from video_processor import VideoProducer
from spark_video_processor import SparkVideoProcessor
from monitoring import MetricsCollector

logger = logging.getLogger('veristream')
metrics = MetricsCollector(port=8000)

class VideoAnalyzer:
    def __init__(self):
        self.whisper_model, self.detector = load_models()
        self.producer = VideoProducer()
        self.spark_processor = SparkVideoProcessor()
        self.metrics = metrics

    def process_frame_batch(self, frames: List[np.ndarray]) -> Tuple[List[float], int]:
        try:
            start_time = time.time()
            scores = self.detector.predict_batch(frames)
            processing_time = time.time() - start_time

            self.metrics.record_metric('processing_time', float(processing_time))
            self.metrics.record_metric('frames_processed', float(len(frames)))

            return scores, len(frames)
        except Exception as e:
            logger.error(f"Error processing frame batch: {e}", exc_info=True)
            self.metrics.system_healthy.set(0)
            return [], 0 

    def analyze_video(self, video_path: str, progress_bar) -> Tuple[str, float, Dict]:
        try:
            cap = cv2.VideoCapture(video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))

            frames_data = {
                'scores': [],
                'max_scores': [],
                'timestamps': [],
                'faces_detected': []  
            }

            batch_size = 32
            frames_batch = []
            frame_count = 0
            all_max_scores = []

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frames_batch.append(frame)
                frame_count += 1

                if len(frames_batch) >= batch_size:
                    scores, num_frames = self.process_frame_batch(frames_batch) 

                    for i, score in enumerate(scores):
                        frames_data['scores'].append(score)
                        frames_data['max_scores'].append(score)
                        timestamp = (frame_count - len(frames_batch) + i) / fps
                        frames_data['timestamps'].append(float(timestamp))
                        frames_data['faces_detected'].append(1 if score > 0 else 0)  
                    all_max_scores.extend(scores)
                    frames_batch = []
                    progress_bar.progress(min(frame_count / total_frames, 0.5))

            cap.release()
            if frames_batch:
                scores, num_frames = self.process_frame_batch(frames_batch) 
                for i, score in enumerate(scores):
                    frames_data['scores'].append(score)
                    frames_data['max_scores'].append(score)
                    timestamp = (frame_count - len(frames_batch) + i) / fps
                    frames_data['timestamps'].append(float(timestamp))
                    frames_data['faces_detected'].append(1 if score > 0 else 0) 
                all_max_scores.extend(scores)

            progress_bar.progress(0.7)
            streaming_query = self.spark_processor.start_streaming(self.detector)  
            transcription = self.whisper_model.transcribe(video_path)
            self.producer.send_video(video_path)  

            final_score = float(np.mean(all_max_scores)) if all_max_scores else 0.0

            return transcription["text"], final_score, frames_data

        except Exception as e:
            logger.error(f"Error analyzing video: {e}", exc_info=True)
            self.metrics.system_healthy.set(0)
            raise