# backend/video_analyzer.py
import cv2
import logging
import whisper
import os
import subprocess
import numpy as np
from typing import List, Dict, Tuple, Any # Added Tuple, Any
# Import detector for internal use in this class's analyze_video method
from optimized_deepfake_detector import OptimizedDeepfakeDetector

logger = logging.getLogger(__name__)

class VideoAnalyzer:
    def __init__(self):
        """Initializes Whisper model and an internal DeepfakeDetector instance."""
        try:
            # Load Whisper model (consider making model size configurable)
            # Options: tiny, base, small, medium, large
            # 'base' is a good balance for performance/accuracy
            self.whisper_model = whisper.load_model("base")
            logger.info("Whisper model (base) loaded for VideoAnalyzer.")
            # This instance is used specifically when analyze_video is called (e.g., for uploads)
            self.internal_deepfake_detector = OptimizedDeepfakeDetector()
            logger.info("Internal DeepfakeDetector instance created for VideoAnalyzer.")
        except Exception as e:
            logger.error(f"Failed to initialize VideoAnalyzer components: {e}", exc_info=True)
            raise RuntimeError("VideoAnalyzer initialization failed") from e


    def analyze_video(self, video_path: str, progress_callback=None) -> Tuple[str, float, Dict[str, Any], str]:
        """
        Analyzes a video file for transcription and deepfake score.

        Args:
            video_path: Path to the video file.
            progress_callback: Optional function to report progress (0.0 to 1.0).

        Returns:
            A tuple containing:
            - transcription (str): The transcribed text.
            - final_score (float): The highest deepfake score found.
            - frames_data (Dict): Dict with 'timestamps', 'max_scores', 'faces_detected'.
            - detected_language (str): The language code detected by Whisper.
        """
        logger.info(f"Starting video analysis for: {video_path}")
        if not os.path.exists(video_path):
             logger.error(f"Video file not found: {video_path}")
             raise FileNotFoundError(f"Video file not found: {video_path}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"Failed to open video file: {video_path}")
            raise ValueError(f"Failed to open video file: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        logger.info(f"Total frames to process: {total_frames}")
        frames = []
        timestamps = []
        processed_frame_count = 0

        # 1. Collect all frames and timestamps
        logger.debug("Collecting frames...")
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # Basic check for valid frame
            if frame is None or frame.size == 0:
                logger.warning(f"Read invalid frame at index {processed_frame_count}, skipping.")
                continue

            frames.append(frame) # Store original frame for deepfake analysis
            timestamps.append(cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0)
            processed_frame_count += 1
            if progress_callback and total_frames > 0 and processed_frame_count % 30 == 0: # Update progress roughly every second
                progress_callback(0.1 + 0.3 * (processed_frame_count / total_frames)) # Progress for frame reading (10%-40%)

        cap.release()
        logger.info(f"Collected {len(frames)} frames.")
        if not frames:
            logger.warning(f"No valid frames collected from {video_path}")
            # Return default values if no frames
            return "No Video Frames", 0.0, {"timestamps": [], "max_scores": [], "faces_detected": []}, "unknown"

        if progress_callback: progress_callback(0.4) # Frame reading done

        # 2. Extract audio and transcribe
        logger.debug("Extracting and transcribing audio...")
        audio_path = os.path.join(os.path.dirname(video_path), f"temp_audio_{os.path.basename(video_path)}.wav")
        transcription = "[Transcription Failed]"
        detected_language = "unknown"
        try:
            # Extract audio using ffmpeg
            cmd_audio = [ "ffmpeg", "-hide_banner", "-loglevel", "error",
                "-i", video_path, "-vn", # No video
                "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", # Standard WAV settings for Whisper
                audio_path, "-y" ]
            subprocess.run(cmd_audio, check=True, capture_output=True) # Use check=True, capture output

            if os.path.exists(audio_path) and os.path.getsize(audio_path) > 44:
                # Transcribe audio with Whisper (using the instance's model)
                logger.info(f"Transcribing audio from {video_path}")
                result = self.whisper_model.transcribe(audio_path, fp16=False) # Use fp32 on CPU
                transcription = result["text"].strip()
                detected_language = result["language"]
                logger.info(f"Transcription complete. Language: {detected_language}")
            else:
                 logger.warning(f"Audio file empty or not created: {audio_path}")
                 transcription = "[No Audio Data]"

        except subprocess.CalledProcessError as e:
            logger.error(f"FFmpeg audio extraction failed: {e.stderr.decode()}")
        except Exception as e:
            logger.error(f"Failed to transcribe audio: {str(e)}", exc_info=True)
        finally:
            if os.path.exists(audio_path):
                try: os.remove(audio_path)
                except Exception as e_del: logger.warning(f"Could not remove temp audio {audio_path}: {e_del}")

        if progress_callback: progress_callback(0.6) # Transcription done

        # 3. Batch predict deepfake scores
        logger.debug(f"Performing deepfake detection on {len(frames)} frames...")
        # Use the internal detector instance for this class
        # Assuming predict_batch expects a list of numpy arrays (frames)
        scores = self.internal_deepfake_detector.predict_batch(frames)
        if len(scores) != len(frames):
             logger.warning(f"Deepfake score count ({len(scores)}) doesn't match frame count ({len(frames)}). Padding with 0.0.")
             # Pad scores if mismatch (e.g., due to preprocessing errors)
             padded_scores = ([0.0] * len(frames))
             for i in range(min(len(scores), len(frames))):
                 padded_scores[i] = scores[i]
             scores = padded_scores

        final_score = max(scores) if scores else 0.0
        logger.info(f"Deepfake detection complete. Max score: {final_score:.3f}")
        if progress_callback: progress_callback(0.9) # Deepfake done

        # 4. Prepare results (Placeholder for face detection)
        # If you need actual face detection, integrate OpenCV Haar Cascade or MTCNN here
        # For now, using a placeholder
        faces_detected_placeholder = [False] * len(frames) # Example: Assume no faces detected

        frames_data = {
            "timestamps": timestamps,
            "max_scores": scores, # List of scores per frame
            "faces_detected": faces_detected_placeholder # Placeholder list
        }

        logger.info(f"Video analysis finished for: {video_path}")
        return transcription, final_score, frames_data, detected_language