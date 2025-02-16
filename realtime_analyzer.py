# realtime_analyzer.py
import cv2
import time
import threading
import numpy as np
import queue
from audio_processor import AudioProcessor
from analyzer import OptimizedAnalyzer
from video_processor import VideoProducer
from utils import display_realtime_results
from models import load_models
import logging

logger = logging.getLogger('realtime')

class RealTimeAnalyzer:
    def __init__(self):
        self.whisper_model, self.detector = load_models()
        self.text_analyzer = OptimizedAnalyzer(use_gpu=True)
        self.audio_processor = AudioProcessor()
        self.producer = VideoProducer()
        self.frame_queue = queue.Queue(maxsize=2000)
        self.last_frame_time = None
        self.last_deepfake_check = time.time() 
        self.results = {}
        self.running = False
        self.last_update = time.time()

    def start_audio_capture(self):
        """Start capturing and processing audio in real-time"""
        def audio_capture_thread():
            # In realtime_analyzer.py
            def audio_capture_thread():
                self.audio_processor.start_stream()
                while self.running:
                    audio_chunk = self.audio_processor.get_chunk()
                    if audio_chunk is not None and len(audio_chunk) > 0:
                        try:
                            text = self.whisper_model.transcribe(audio_chunk)["text"]
                            if text:
                                analysis = self.text_analyzer.analyze_text(text)
                                self.results['text_analysis'] = analysis
                        except Exception as e:
                            logger.error(f"Audio processing error: {e}")
                    time.sleep(0.1)

        threading.Thread(target=audio_capture_thread, daemon=True).start()

    def start_video_capture(self):
        """Start capturing and processing video frames in real-time"""
        def video_capture_thread():
            cap = cv2.VideoCapture(0)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            
            while self.running and cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    continue
                
                # Process frame
                processed_frame = self.process_frame(frame)
                
                # Send to Kafka
                self.producer.send_frame(frame)
                
                # Add to display queue
                if self.frame_queue.full():
                    self.frame_queue.get()
                self.frame_queue.put(processed_frame)
                
            cap.release()

        threading.Thread(target=video_capture_thread, daemon=True).start()

# In realtime_analyzer.py
    def process_frame(self, frame):
        try:
            start_time = time.time()
            current_time = time.time()

        # Check if it's time to update the score (every 20 seconds)
            if current_time - self.last_score_update_time >= self.score_update_interval:
                avg_score, max_score = self.detector.predict_batch([frame])
                processing_time = time.time() - start_time

            # Update last score update time
                self.last_score_update_time = current_time

                self.results['video_metrics'] = {
                    'latest_score': max_score, # Update the score
                    'processing_time': processing_time,
                    'frame_rate': 1/processing_time if processing_time > 0 else 0
                }
            else:
            # If not time for score update, just record processing time (minimal)
                processing_time = time.time() - start_time # Still track processing for frame display
                if 'video_metrics' in self.results: # Keep previous score
                    self.results['video_metrics']['processing_time'] = processing_time
                else: # Initialize metrics if not already done (first frame before 20s)
                    self.results['video_metrics'] = {
                        'latest_score': 0.0, # Or some default value
                        'processing_time': processing_time,
                        'frame_rate': 1/processing_time if processing_time > 0 else 0
                    }


        # Convert to RGB for display (this part always happens)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Add overlay (this part always happens)
            if 'video_metrics' in self.results and 'latest_score' in self.results['video_metrics']: # Check if metrics are available
                label = f"Deepfake: {self.results['video_metrics']['latest_score']:.2%}"
                color = (255, 0, 0) if self.results['video_metrics']['latest_score'] > 0.7 else (0, 255, 0)
                cv2.putText(frame, label, (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)


            return frame
        
        except Exception as e:
            logger.error(f"Frame processing error: {e}")
            return frame


    def start_analysis(self):
        """Start real-time analysis"""
        self.running = True
        self.start_audio_capture()
        self.start_video_capture()
        logger.info("Real-time analysis started")

    def stop_analysis(self):
        """Stop real-time analysis"""
        self.running = False
        self.audio_processor.stop_stream()
        self.producer.close()
        logger.info("Real-time analysis stopped")

    def get_latest_frame(self):
        """Get the latest processed frame for display"""
        try:
            return self.frame_queue.get_nowait()
        except queue.Empty:
            return None
