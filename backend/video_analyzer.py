# backend/video_analyzer.py
import cv2
import logging
import whisper
import os
import subprocess
# Use the updated detector class
from optimized_deepfake_detector import OptimizedDeepfakeDetector
import numpy as np
import time
# Use a common temp directory definition if possible, or define locally
TEMP_DIR = "temp_media" # Make sure this matches main.py StaticFiles mount
# Ensure the temp directory exists (can be created here or in main.py)
if not os.path.exists(TEMP_DIR):
    try:
        os.makedirs(TEMP_DIR)
    except OSError as e:
        # Handle potential race condition if another process creates it
        if not os.path.isdir(TEMP_DIR):
            raise e

logger = logging.getLogger(__name__) # Standard logger setup

class VideoAnalyzer:
    def __init__(self):
        """Initializes components needed for video analysis."""
        # Instantiate detector here if VideoAnalyzer is self-contained
        # Or potentially accept it as an argument if managed globally in main.py
        self.deepfake_detector = OptimizedDeepfakeDetector()
        try:
            # Consider loading model size based on environment/config
            self.whisper_model = whisper.load_model("base")
            logger.info("Whisper model loaded.")
        except Exception as e:
            logger.error(f"Failed to load Whisper model: {e}", exc_info=True)
            # Decide how to handle - raise error or proceed without transcription?
            self.whisper_model = None # Set to None if loading failed
            # raise RuntimeError("Whisper model failed to load") from e
        logger.info("VideoAnalyzer initialized.")

    def analyze_video(self, video_path: str, progress_callback=None, num_explanations=3):
        """
        Analyzes an uploaded video file.

        Args:
            video_path (str): Path to the video file.
            progress_callback: Optional callback for progress updates (not used here).
            num_explanations (int): Number of top-scoring frames to generate explanations for.

        Returns:
            tuple: (transcription, final_score, frames_data, detected_language)
                   where frames_data includes scores, timestamps, and explanation image URLs.
        """
        logger.info(f"Starting analysis for video: {video_path}")
        frames = []
        timestamps = []
        cap = None

        # --- 1. Read Frames ---
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError(f"Failed to open video file: {video_path}")

            while True:
                ret, frame = cap.read()
                if not ret: break
                # Store frames as NumPy arrays (BGR)
                frames.append(frame)
                timestamps.append(cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0)

            if not frames:
                 raise ValueError(f"No frames could be read from video: {video_path}")
            logger.info(f"Read {len(frames)} frames from video.")

        except Exception as e:
            logger.error(f"Error reading video frames: {e}", exc_info=True)
            # Return default/error values matching the expected 4-tuple return type
            return "[Frame Reading Failed]", 0.0, {"timestamps": [], "max_scores": [], "faces_detected": [], "explanations": []}, "unknown"
        finally:
            if cap: cap.release()

        # --- 2. Deepfake Scoring ---
        scores = []
        final_score = 0.0
        try:
            logger.info(f"Calculating deepfake scores...")
            # predict_batch handles preprocessing and returns scores for each input frame
            scores = self.deepfake_detector.predict_batch(frames)
            final_score = max(scores) if scores else 0.0
            logger.info(f"Deepfake scoring complete. Max score: {final_score:.3f}")
        except Exception as e_pred:
             logger.error(f"Error during deepfake prediction: {e_pred}", exc_info=True)
             # Assign default scores if prediction fails
             scores = [0.0] * len(frames)

        # Prepare initial frames_data (ensure lists match length of original frames)
        frames_data = {
            "timestamps": timestamps[:len(frames)],
            "max_scores": scores[:len(frames)],
            "faces_detected": [True] * len(frames), # Placeholder - update if face detection added
            "explanations": [] # Initialize explanations list
        }

        # --- 3. Explanation Generation (New) ---
        explanation_images = []
        if scores and num_explanations > 0:
            try:
                # Find indices of frames with the highest scores
                sorted_indices = np.argsort(scores)[::-1] # Indices sorted by score desc
                top_indices = sorted_indices[:min(num_explanations, len(scores))]

                logger.info(f"Generating explanations for top {len(top_indices)} scoring frames...")
                for frame_idx in top_indices:
                    if frame_idx >= len(frames): continue # Safety check

                    frame_to_explain = frames[frame_idx]
                    score = scores[frame_idx]
                    timestamp = timestamps[frame_idx]
                    logger.debug(f"Explaining frame index {frame_idx} (score: {score:.3f}, time: {timestamp:.2f}s)")

                    # Generate heatmap using the detector instance
                    heatmap = self.deepfake_detector.get_attention_maps(frame_to_explain)

                    if heatmap is not None:
                        # Create overlay using the detector instance
                        overlay = self.deepfake_detector.create_overlay(frame_to_explain, heatmap, alpha=0.6)

                        if overlay is not None:
                            # Save overlay image to TEMP_DIR
                            base_filename = os.path.splitext(os.path.basename(video_path))[0]
                            # Sanitize base_filename if needed
                            safe_base_filename = "".join(c if c.isalnum() else "_" for c in base_filename)
                            explanation_filename = f"explain_{safe_base_filename}_idx{frame_idx}_ts{timestamp:.1f}.jpg"
                            explanation_path = os.path.join(TEMP_DIR, explanation_filename)

                            try:
                                success = cv2.imwrite(explanation_path, overlay)
                                if success:
                                    # Store RELATIVE URL for frontend
                                    explanation_url = f"/{TEMP_DIR}/{explanation_filename}"
                                    explanation_images.append({
                                        "frame_index": int(frame_idx), # Ensure integer
                                        "timestamp": float(timestamp),
                                        "score": float(score),
                                        "url": explanation_url
                                    })
                                    logger.debug(f"Saved explanation image: {explanation_path}")
                                else:
                                    logger.warning(f"Failed to save explanation image (cv2.imwrite returned False) for frame {frame_idx}")
                            except Exception as e_save:
                                logger.error(f"Error saving explanation image {explanation_path}: {e_save}")
                        else:
                            logger.warning(f"Failed to create overlay for frame {frame_idx}")
                    else:
                        logger.warning(f"Failed to generate heatmap for frame {frame_idx}")

            except Exception as e_explain:
                 logger.error(f"Error during explanation generation: {e_explain}", exc_info=True)

        # Add generated explanation info to frames_data
        frames_data["explanations"] = explanation_images

        # --- 4. Transcription ---
        transcription = "[Transcription Skipped]"
        detected_language = "unknown"
        if self.whisper_model: # Check if model loaded successfully
            # Unique audio path within TEMP_DIR
            safe_video_basename = "".join(c if c.isalnum() else "_" for c in os.path.splitext(os.path.basename(video_path))[0])
            audio_path = os.path.join(TEMP_DIR, f"audio_{safe_video_basename}_{int(time.time())}.wav")
            try:
                logger.info(f"Extracting audio to {audio_path}...")
                cmd_audio = [
                    "ffmpeg", "-hide_banner", "-loglevel", "warning",
                    "-i", video_path,
                    "-vn", # No video
                    "-acodec", "pcm_s16le", # Standard WAV codec
                    "-ar", "16000", # Whisper preferred sample rate
                    "-ac", "1", # Mono audio
                    audio_path, "-y"
                ]
                # Use check=True and capture output for better error diagnosis
                result = subprocess.run(cmd_audio, check=True, capture_output=True, text=True, encoding='utf-8')

                if os.path.exists(audio_path) and os.path.getsize(audio_path) > 44: # Check > WAV header size
                    logger.info(f"Transcribing audio file: {audio_path}")
                    # Note: whisper transcribe can be slow, run in thread in main.py
                    result = self.whisper_model.transcribe(audio_path)
                    transcription = result["text"]
                    detected_language = result["language"]
                    logger.info(f"Transcription complete. Language: {detected_language}")
                else:
                    logger.warning(f"Audio file not generated or empty: {audio_path}")
                    transcription = "[No Audio Data]"
            except subprocess.CalledProcessError as e_ffmpeg:
                 logger.error(f"FFmpeg audio extraction failed: {e_ffmpeg.stderr}")
                 transcription = "[Audio Extraction Failed]"
            except Exception as e_audio:
                logger.error(f"Error in audio processing/transcription: {str(e_audio)}", exc_info=True)
                transcription = "[Transcription Error]"
            finally:
                # Cleanup temporary audio file
                if os.path.exists(audio_path):
                    try: os.remove(audio_path)
                    except Exception as e_del: logger.warning(f"Failed to delete temp audio {audio_path}: {e_del}")
        else:
             logger.warning("Whisper model not loaded, skipping transcription.")


        # --- 5. Return Results ---
        logger.info(f"Analysis function finished for {video_path}.")
        # frames_data now contains explanations
        return transcription, final_score, frames_data, detected_language