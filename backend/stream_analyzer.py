# backend/stream_analyzer.py
import cv2
import numpy as np
import asyncio
import logging
import subprocess
from streamlink import Streamlink
from optimized_deepfake_detector import OptimizedDeepfakeDetector
from video_analyzer import VideoAnalyzer
from analyzer import OptimizedAnalyzer
import queue
import threading
import time
import os

logger = logging.getLogger(__name__)

class StreamAnalyzer:
    def __init__(self):
        self.detector = OptimizedDeepfakeDetector()
        self.video_analyzer = VideoAnalyzer()
        self.text_analyzer = OptimizedAnalyzer(use_gpu=True)
        self.session = Streamlink()
        self.is_running = False
        self.frame_queue = queue.Queue(maxsize=30)
        self.latest_result = None
        self.stream_thread = None
        self.audio_buffer = []

    async def read_stream(self, stream_url):
        """Read stream frames and audio using Streamlink and FFmpeg."""
        logger.info(f"Starting stream analysis for {stream_url}")
        while self.is_running:
            try:
                streams = self.session.streams(stream_url)
                if not streams:
                    logger.error(f"No streams found for {stream_url}")
                    raise ValueError("No streams available")
                
                stream = streams.get("best")
                if not stream:
                    raise ValueError("Best stream not available")
                
                fd = stream.open()
                frame_count = 0
                start_time = time.time()
                transcription_chunk = ""

                while self.is_running:
                    # Capture 1-second chunk with FFmpeg directly
                    temp_video = f"temp_stream_{frame_count}.mp4"
                    temp_audio = f"temp_audio_{frame_count}.wav"
                    cmd = [
                        "ffmpeg", "-i", fd.url if hasattr(fd, 'url') else stream.url,
                        "-t", "1",  # 1-second duration
                        "-c:v", "libx264", "-c:a", "aac",  # Video and audio codecs
                        temp_video, "-y"
                    ]
                    result = subprocess.run(cmd, capture_output=True, text=True)
                    if result.returncode != 0:
                        logger.error(f"FFmpeg capture failed: {result.stderr}")
                        break

                    frame_count += 30  # Increment by ~30 frames (assuming 30 FPS)
                    timestamp = (time.time() - start_time)

                    # Read video for deepfake detection
                    cap = cv2.VideoCapture(temp_video)
                    frames = []
                    while cap.isOpened():
                        ret, frame = cap.read()
                        if not ret:
                            break
                        frame = cv2.resize(frame, (640, 480))
                        frames.append(frame)
                    cap.release()

                    if frames:
                        scores = self.detector.predict_batch(frames)
                        score = max(scores) if scores else 0.0
                        try:
                            self.frame_queue.put_nowait({"timestamp": timestamp, "score": score})
                        except queue.Full:
                            self.frame_queue.get_nowait()
                            self.frame_queue.put_nowait({"timestamp": timestamp, "score": score})

                    # Extract audio
                    cmd = [
                        "ffmpeg", "-i", temp_video, "-vn", "-acodec", "pcm_s16le",
                        "-ar", "16000", "-ac", "1", temp_audio, "-y"
                    ]
                    result = subprocess.run(cmd, capture_output=True, text=True)
                    if result.returncode != 0:
                        logger.error(f"FFmpeg audio extraction failed: {result.stderr}")
                        continue

                    # Transcribe
                    if os.path.exists(temp_audio) and os.path.getsize(temp_audio) > 0:
                        result = self.video_analyzer.whisper_model.transcribe(temp_audio)
                        transcription_chunk += " " + result["text"]
                    else:
                        logger.warning("No audio extracted, skipping transcription")
                        transcription_chunk += " [No audio]"

                    # Text analysis
                    text_analysis = await self.text_analyzer.analyze_text(transcription_chunk)

                    # Update latest result
                    self.latest_result = {
                        "transcription": transcription_chunk,
                        "frames": [{"timestamp": timestamp, "score": score}],
                        "textAnalysis": {
                            "sentiment": text_analysis.sentiment,
                            "fact_checks": text_analysis.fact_checks,
                            "emotional_triggers": text_analysis.emotional_triggers,
                            "stereotypes": text_analysis.stereotypes,
                            "manipulation_score": text_analysis.manipulation_score,
                            "entities": text_analysis.entities
                        }
                    }

                    # Cleanup
                    os.remove(temp_video)
                    os.remove(temp_audio)

                    await asyncio.sleep(1)  # Process every second

                fd.close()
                await self.reconnect(stream_url)

            except Exception as e:
                logger.error(f"Stream error: {str(e)}")
                await self.reconnect(stream_url)

    async def reconnect(self, stream_url):
        """Reconnect with exponential backoff."""
        for attempt in range(3):
            wait_time = 2 ** attempt
            logger.info(f"Reconnection attempt {attempt + 1}/3, waiting {wait_time}s")
            await asyncio.sleep(wait_time)
            if self.is_running:
                return
        logger.error("Max reconnection attempts reached")
        self.is_running = False

    def start_analysis(self, stream_url):
        """Start the stream analysis in a separate thread."""
        self.is_running = True
        self.latest_result = None
        if self.stream_thread is None or not self.stream_thread.is_alive():
            self.stream_thread = threading.Thread(
                target=self._run_stream_loop,
                args=(stream_url,),
                daemon=True
            )
            self.stream_thread.start()
            logger.info("Stream analysis thread started")

    def _run_stream_loop(self, stream_url):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(self.read_stream(stream_url))
        loop.close()

    def stop_analysis(self):
        """Stop the stream analysis."""
        self.is_running = False
        if self.stream_thread:
            self.stream_thread.join(timeout=2)
        while not self.frame_queue.empty():
            self.frame_queue.get_nowait()
        logger.info("Stream analysis stopped")

    def get_latest_result(self):
        """Get the latest analysis result."""
        return self.latest_result

if __name__ == "__main__":
    analyzer = StreamAnalyzer()
    analyzer.start_analysis("https://www.twitch.tv/iskall85")
    time.sleep(60)  # Run for 60s
    analyzer.stop_analysis()