# backend/video_analyzer.py
import cv2
import logging
import whisper
import os
import subprocess
import numpy as np
# --- Keep existing imports ---
from typing import List, Dict, Tuple, Any
# --- Import the updated detector ---
from optimized_deepfake_detector import OptimizedDeepfakeDetector
import time # Needed for unique filenames

logger = logging.getLogger(__name__)

# Define a threshold for generating heatmaps
HEATMAP_SCORE_THRESHOLD = 0.7

class VideoAnalyzer:
    def __init__(self):
        """Initializes Whisper model and an internal DeepfakeDetector instance."""
        try:
            self.whisper_model = whisper.load_model("base")
            logger.info("Whisper model (base) loaded for VideoAnalyzer.")
            # This instance is used specifically when analyze_video is called
            self.internal_deepfake_detector = OptimizedDeepfakeDetector()
            logger.info("Internal DeepfakeDetector instance created for VideoAnalyzer.")
            # Ensure TEMP_DIR exists (might be redundant if main.py ensures it)
            self.temp_dir = "temp_media"
            if not os.path.exists(self.temp_dir):
                os.makedirs(self.temp_dir)
        except Exception as e:
            logger.error(f"Failed to initialize VideoAnalyzer components: {e}", exc_info=True)
            raise RuntimeError("VideoAnalyzer initialization failed") from e


    def analyze_video(self, video_path: str, progress_callback=None) -> Tuple[str, float, Dict[str, Any], str]:
        """
        Analyzes a video file for transcription and deepfake score,
        generating heatmap overlays for frames above a threshold.

        Args:
            video_path: Path to the video file.
            progress_callback: Optional function to report progress (0.0 to 1.0).

        Returns:
            A tuple containing:
            - transcription (str): The transcribed text.
            - final_score (float): The highest deepfake score found.
            - frames_data (Dict): Dict with 'timestamps', 'max_scores', 'faces_detected',
                                  and 'overlay_urls' (mapping timestamp index to URL).
            - detected_language (str): The language code detected by Whisper.
        """
        logger.info(f"Starting video analysis with heatmaps for: {video_path}")
        start_time_analysis = time.time() # For unique filenames
        base_filename = f"upload_{int(start_time_analysis)}_{os.path.basename(video_path).split('.')[0]}"

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
        frame_indices = [] # Store original index

        # 1. Collect all frames, timestamps, and indices
        logger.debug("Collecting frames...")
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame is None or frame.size == 0:
                logger.warning(f"Read invalid frame at index {processed_frame_count}, skipping.")
                processed_frame_count += 1 # Still increment index
                continue

            current_index = processed_frame_count
            frames.append(frame) # Store original frame
            timestamps.append(cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0)
            frame_indices.append(current_index) # Keep track of the index

            processed_frame_count += 1
            if progress_callback and total_frames > 0 and processed_frame_count % 30 == 0:
                progress_callback(0.1 + 0.3 * (processed_frame_count / total_frames))

        cap.release()
        logger.info(f"Collected {len(frames)} valid frames (from {processed_frame_count} total reads).")
        if not frames:
            logger.warning(f"No valid frames collected from {video_path}")
            return "No Video Frames", 0.0, {"timestamps": [], "max_scores": [], "faces_detected": [], "overlay_urls": {}}, "unknown"

        if progress_callback: progress_callback(0.4)

        # 2. Extract audio and transcribe
        # (Audio extraction and transcription logic remains the same)
        logger.debug("Extracting and transcribing audio...")
        audio_path = os.path.join(self.temp_dir, f"temp_audio_{base_filename}.wav")
        transcription = "[Transcription Failed]"
        detected_language = "unknown"
        try:
            cmd_audio = [ "ffmpeg", "-hide_banner", "-loglevel", "error",
                "-i", video_path, "-vn",
                "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
                audio_path, "-y" ]
            subprocess.run(cmd_audio, check=True, capture_output=True)

            if os.path.exists(audio_path) and os.path.getsize(audio_path) > 44:
                logger.info(f"Transcribing audio from {video_path}")
                result = self.whisper_model.transcribe(audio_path, fp16=False)
                transcription = result["text"].strip()
                detected_language = result["language"]
                logger.info(f"Transcription complete. Language: {detected_language}")
            else:
                 logger.warning(f"Audio file empty or not created: {audio_path}")
                 transcription = "[No Audio Data]"
        except subprocess.CalledProcessError as e:
            logger.error(f"FFmpeg audio extraction failed: {e.stderr.decode()}")
            transcription = "[Audio Extraction Error]"
        except Exception as e:
            logger.error(f"Failed to transcribe audio: {str(e)}", exc_info=True)
            transcription = "[Transcription Error]"
        finally:
            if os.path.exists(audio_path):
                try: os.remove(audio_path)
                except Exception as e_del: logger.warning(f"Could not remove temp audio {audio_path}: {e_del}")

        if progress_callback: progress_callback(0.6) # Transcription done (approx)

        # 3. Batch predict deepfake scores
        logger.debug(f"Performing deepfake detection on {len(frames)} frames...")
        # This uses the internal detector instance
        scores = self.internal_deepfake_detector.predict_batch(frames)

        # Ensure scores list matches frames list length (predict_batch should handle this now)
        if len(scores) != len(frames):
             logger.error(f"CRITICAL: Score count ({len(scores)}) mismatch with frame count ({len(frames)}) after predict_batch.")
             # Handle error - maybe return or pad cautiously? Padding might misalign scores. Best to error out?
             # For now, let's pad with 0.0 but log loudly.
             padded_scores = ([0.0] * len(frames))
             count = min(len(scores), len(frames))
             padded_scores[:count] = scores[:count]
             scores = padded_scores
             # Recompute final_score based on potentially padded scores
             final_score = max(scores) if scores else 0.0
        else:
            final_score = max(scores) if scores else 0.0

        logger.info(f"Deepfake detection complete. Max score: {final_score:.3f}")
        if progress_callback: progress_callback(0.8) # Deepfake scoring done

        # 4. Generate Heatmaps/Overlays for high-scoring frames
        logger.debug(f"Generating heatmap overlays for scores > {HEATMAP_SCORE_THRESHOLD}...")
        overlay_urls = {} # Dictionary to store {frame_index: url}
        heatmap_generated_count = 0
        for i in range(len(frames)):
            score = scores[i]
            if score > HEATMAP_SCORE_THRESHOLD:
                frame_index = frame_indices[i] # Get original index
                frame = frames[i]
                logger.info(f"Generating heatmap for frame {frame_index} (score: {score:.3f})")
                try:
                    heatmap = self.internal_deepfake_detector.get_attention_maps(frame)
                    if heatmap is not None:
                        overlay = self.internal_deepfake_detector.create_overlay(frame, heatmap)
                        if overlay is not None:
                            # Save overlay image
                            overlay_filename = f"{base_filename}_frame_{frame_index}_overlay.jpg"
                            overlay_path = os.path.join(self.temp_dir, overlay_filename)
                            # Use a different variable for save success
                            save_success = cv2.imwrite(overlay_path, overlay)
                            if save_success:
                                overlay_url = f"/{self.temp_dir}/{overlay_filename}" # Relative URL for serving
                                overlay_urls[frame_index] = overlay_url # Store URL mapped by original frame index
                                heatmap_generated_count += 1
                                logger.debug(f"Saved overlay for frame {frame_index} to {overlay_path}")
                            else:
                                 logger.warning(f"Failed to save overlay image for frame {frame_index} using cv2.imwrite.")
                        else:
                            logger.warning(f"Overlay creation failed for frame {frame_index}")
                    else:
                        logger.warning(f"Heatmap generation failed for frame {frame_index}")
                except Exception as e_overlay:
                    logger.error(f"Error generating/saving overlay for frame {frame_index}: {e_overlay}", exc_info=True)

            # Update progress during heatmap generation (can be slow if many frames are above threshold)
            if progress_callback and total_frames > 0 and i % 10 == 0: # Update less frequently
                progress = 0.8 + 0.15 * (i / len(frames)) # Progress for overlay generation (80%-95%)
                progress_callback(progress)

        logger.info(f"Generated {heatmap_generated_count} heatmap overlays.")
        if progress_callback: progress_callback(0.95) # Overlays done

        # 5. Prepare final results structure
        # Placeholder for face detection (as before)
        faces_detected_placeholder = [False] * len(frames)

        frames_data = {
            "timestamps": timestamps, # Corresponds to collected frames
            "scores": scores, # Corresponds to collected frames
            "frame_indices": frame_indices, # Original indices of collected frames
            "faces_detected": faces_detected_placeholder, # Placeholder
            "overlay_urls": overlay_urls # Dict: {original_frame_index: url}
        }
        # RENAME 'max_scores' to 'scores' to be clearer it's per-frame
        # 'final_score' is the overall max

        logger.info(f"Video analysis finished for: {video_path}")
        if progress_callback: progress_callback(1.0) # Final step

        # Return updated structure
        return transcription, final_score, frames_data, detected_language