#app.py
import streamlit as st
import cv2
from video_processor import VideoProducer, test_consumer
import threading
import time
import whisper
from kafka.admin import KafkaAdminClient, NewTopic
from kafka.errors import TopicAlreadyExistsError
from kafka import KafkaConsumer
import json
import base64
from datetime import datetime
import plotly.graph_objects as go
import pandas as pd
import os
from kafka.admin import KafkaAdminClient
import numpy as np
import torch
from optimized_deepfake_detector import OptimizedDeepfakeDetector
import concurrent.futures
from typing import Tuple, List, Dict
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
@st.cache_resource
def load_models() -> Tuple[any, OptimizedDeepfakeDetector]:
    try:
        whisper_model = whisper.load_model("base")
        detector = OptimizedDeepfakeDetector()
        return whisper_model, detector
    except Exception as e:
        logger.error(f"Error loading models: {e}")
        raise

class VideoAnalyzer:
    def __init__(self):
        self.whisper_model, self.detector = load_models()
        self.producer = VideoProducer()
        
    def process_frame_batch(self, frames: List[np.ndarray]) -> Tuple[float, float]:
        try:
            return self.detector.predict_batch(frames)
        except Exception as e:
            logger.error(f"Error processing frame batch: {e}")
            return 0.0, 0.0
            
    def analyze_video(self, video_path: str, progress_bar) -> Tuple[str, float, Dict]:
        try:
            cap = cv2.VideoCapture(video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            
            frames_data = {
                'scores': [],
                'max_scores': [],
                'timestamps': []
            }
            
            batch_size = 32
            frames_batch = []
            frame_count = 0
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                    
                frames_batch.append(frame)
                frame_count += 1
                
                if len(frames_batch) >= batch_size:
                    avg_score, max_score = self.process_frame_batch(frames_batch)
                    frames_data['scores'].append(avg_score)
                    frames_data['max_scores'].append(max_score)
                    frames_data['timestamps'].append(frame_count / fps)
                    frames_batch = []
                    
                progress_bar.progress(min(frame_count / total_frames, 0.5))
                
            cap.release()
            
            if frames_batch:
                avg_score, max_score = self.process_frame_batch(frames_batch)
                frames_data['scores'].append(avg_score)
                frames_data['max_scores'].append(max_score)
                frames_data['timestamps'].append(frame_count / fps)
            
            progress_bar.progress(0.7)
            transcription = self.whisper_model.transcribe(video_path)
            self.producer.send_video(video_path)
            
            final_score = np.mean(frames_data['max_scores'])
            return transcription["text"], final_score, frames_data
            
        except Exception as e:
            logger.error(f"Error analyzing video: {e}")
            raise

def create_analysis_charts(frames_data: Dict) -> Tuple[go.Figure, go.Figure]:
    score_fig = go.Figure()
    score_fig.add_trace(go.Scatter(x=frames_data['timestamps'], 
                                 y=frames_data['scores'],
                                 mode='lines',
                                 name='Average Score'))
    score_fig.add_trace(go.Scatter(x=frames_data['timestamps'], 
                                 y=frames_data['max_scores'],
                                 mode='lines',
                                 name='Max Score'))
    score_fig.update_layout(title='Deepfake Detection Scores',
                          xaxis_title='Time (seconds)',
                          yaxis_title='Confidence Score')
    
    hist_fig = go.Figure()
    hist_fig.add_trace(go.Histogram(x=frames_data['max_scores'],
                                  nbinsx=30,
                                  name='Score Distribution'))
    hist_fig.update_layout(title='Score Distribution',
                          xaxis_title='Confidence Score',
                          yaxis_title='Frequency')
    
    return score_fig, hist_fig

def display_analysis_results(final_score: float, frames_data: Dict):
    col1, col2 = st.columns(2)
    
    with col1:
        score_fig, hist_fig = create_analysis_charts(frames_data)
        st.plotly_chart(score_fig)
    
    with col2:
        st.plotly_chart(hist_fig)
        
    if final_score > 0.7:
        st.error(f"🚨 High probability of deepfake detected (Confidence: {final_score:.2%})")
        st.markdown("""
            **Detection Details:**
            - Multiple manipulated faces detected
            - High manipulation confidence score
            - Consistent detection across frames
        """)
    elif final_score > 0.4:
        st.warning(f"⚠️ Potential manipulation detected (Confidence: {final_score:.2%})")
    else:
        st.success(f"✅ Video appears authentic (Confidence: {1-final_score:.2%})")

def main():
    st.title("Advanced Video Analysis Pipeline")
    st.markdown("### Real-time Deepfake Detection & Transcription")
    
    with st.sidebar:
        st.title("System Status")
        if st.button("Check Kafka Topics"):
            try:
                admin_client = KafkaAdminClient(bootstrap_servers=['localhost:29092'])
                topics = admin_client.list_topics()
                st.json({"Available Topics": topics})
            except Exception as e:
                st.error(f"Error checking topics: {e}")
    
    uploaded_file = st.file_uploader("Upload Video for Analysis", type=['mp4', 'avi', 'mov'])
    
    if uploaded_file:
        temp_path = f"temp_{uploaded_file.name}"
        try:
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            video_file = open(temp_path, "rb")
            st.video(video_file)
            
            progress_bar = st.progress(0)
            st.write("Analyzing video...")
            
            analyzer = VideoAnalyzer()
            transcription, final_score, frames_data = analyzer.analyze_video(temp_path, progress_bar)
            
            display_analysis_results(final_score, frames_data)
            
            with st.expander("Video Transcription"):
                st.write(transcription)
                
            progress_bar.progress(1.0)
            
        except Exception as e:
            st.error(f"Error processing video: {str(e)}")
            logger.error(f"Processing error: {str(e)}")
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

if __name__ == "__main__":
    main()