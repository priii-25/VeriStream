# backend/video_analyzer.py
import cv2
import logging
import whisper
import os
import subprocess
from optimized_deepfake_detector import OptimizedDeepfakeDetector

logger = logging.getLogger(__name__)

class VideoAnalyzer:
    def __init__(self):
        self.deepfake_detector = OptimizedDeepfakeDetector()
        self.whisper_model = whisper.load_model("base")  # Options: tiny, base, small, medium, large
        logger.info("VideoAnalyzer initialized with Whisper model")

    def analyze_video(self, video_path: str, progress_bar=None):
        cap = cv2.VideoCapture(video_path)
        frame_count = 0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frames = []
        timestamps = []
        faces_detected = []

        # Collect frames
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1
            frames.append(frame)
            timestamps.append(cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0)
            faces_detected.append(True)  # Placeholder; adjust as needed

        cap.release()

        # Extract audio and transcribe
        audio_path = "temp_audio.wav"
        try:
            # Extract audio using ffmpeg
            subprocess.run([
                "ffmpeg", "-i", video_path, "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", audio_path,
                "-y"  # Overwrite output if exists
            ], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            # Transcribe audio with Whisper
            logger.info(f"Transcribing audio from {video_path}")
            result = self.whisper_model.transcribe(audio_path)
            transcription = result["text"]
        except Exception as e:
            logger.error(f"Failed to transcribe audio: {str(e)}")
            transcription = "Transcription failed"
        finally:
            if os.path.exists(audio_path):
                os.remove(audio_path)

        # Batch predict deepfake scores
        scores = self.deepfake_detector.predict_batch(frames)
        final_score = max(scores) if scores else 0.0
        frames_data = {
            "timestamps": timestamps,
            "max_scores": scores,
            "faces_detected": faces_detected
        }
        
        return transcription, final_score, frames_data