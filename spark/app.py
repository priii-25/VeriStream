# app.py
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

@st.cache_resource
def load_whisper_model():
    return whisper.load_model("base")

def get_kafka_metrics():
    """Get metrics from Kafka topics"""
    consumer = KafkaConsumer(
        'video-frames',
        bootstrap_servers=['localhost:29092'],
        group_id='metrics-group'
    )
    
    metrics = consumer.metrics()
    consumer.close()
    return metrics

def create_metrics_chart(metrics_history):
    """Create a line chart for Kafka metrics"""
    df = pd.DataFrame(metrics_history)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['timestamp'], y=df['messages'],
                            mode='lines+markers',
                            name='Messages Processed'))
    
    fig.update_layout(
        title='Kafka Messages Processing Rate',
        xaxis_title='Time',
        yaxis_title='Messages',
        height=400
    )
    
    return fig

def process_uploaded_video(video_path, progress_bar):
    """Process uploaded video and return transcription"""
    producer = VideoProducer()
    model = load_whisper_model()
    
    video_file = open(video_path, 'rb')
    video_bytes = video_file.read()
    
    st.video(video_bytes)
    
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    
    result = model.transcribe(video_path)
    transcription = result["text"]
    
    producer.send_video(video_path)
    progress_bar.progress(100)
    
    return transcription

def main():
    st.title("Video Processing Pipeline")
    
    st.sidebar.title("Kafka Monitoring")
    if st.sidebar.button("Refresh Metrics"):
        metrics = get_kafka_metrics()
        st.sidebar.json(metrics)
    
    st.header("Upload Video")
    uploaded_file = st.file_uploader("Choose a video file", type=['mp4', 'avi', 'mov'])
    
    if uploaded_file is not None:
        temp_path = f"temp_{uploaded_file.name}"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        progress_bar = st.progress(0)
        st.write("Processing video...")

        try:
            transcription = process_uploaded_video(temp_path, progress_bar)
            
            st.header("Transcription")
            st.write(transcription)
            
            metrics_history = []
            metrics_placeholder = st.empty()
            
            for i in range(5):
                metrics = get_kafka_metrics()
                metrics_history.append({
                    'timestamp': datetime.now(),
                    'messages': len(metrics)
                })
                
                if len(metrics_history) > 1:
                    chart = create_metrics_chart(metrics_history)
                    metrics_placeholder.plotly_chart(chart)
                
                time.sleep(2)
                
        except Exception as e:
            st.error(f"Error processing video: {e}")
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
    
    st.header("Kafka Topics Status")
    if st.button("Check Topics"):
        try:
            admin_client = KafkaAdminClient(bootstrap_servers=['localhost:29092'])
            topics = admin_client.list_topics()
            st.json({"Available Topics": topics})
        except Exception as e:
            st.error(f"Error fetching topics: {e}")

if __name__ == "__main__":
    main()