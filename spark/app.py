import streamlit as st
import cv2
from video_processor import VideoProducer
from spark_video_processor import SparkVideoProcessor, SparkTranscriptionProcessor
import threading
import time
import whisper
from kafka.admin import KafkaAdminClient, NewTopic
from kafka.errors import TopicAlreadyExistsError
import json
import pandas as pd
import os
import numpy as np
import torch
from optimized_deepfake_detector import OptimizedDeepfakeDetector
import plotly.graph_objects as go
from datetime import datetime, timedelta
import logging
import psutil
import altair as alt
from prometheus_client import Counter, Gauge, Histogram, start_http_server, REGISTRY, CollectorRegistry
from typing import Dict, Any, Optional, Tuple, List
from monitoring import MetricsCollector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize metrics collector
metrics = MetricsCollector(port=8000)

@st.cache_resource
def load_models() -> Tuple[Any, OptimizedDeepfakeDetector]:
    try:
        whisper_model = whisper.load_model("base")
        detector = OptimizedDeepfakeDetector()
        return whisper_model, detector
    except Exception as e:
        logger.error(f"Error loading models: {e}")
        metrics.system_healthy.set(0)
        raise

class VideoAnalyzer:
    def __init__(self):
        self.whisper_model, self.detector = load_models()
        self.producer = VideoProducer()
        self.spark_processor = SparkVideoProcessor()
        self.metrics = metrics
        
    def process_frame_batch(self, frames: List[np.ndarray]) -> Tuple[float, float, int]:
        try:
            start_time = time.time()
            avg_score, max_score = self.detector.predict_batch(frames)
            processing_time = time.time() - start_time
            
            # Record metrics
            self.metrics.record_metric('processing_time', float(processing_time))
            self.metrics.record_metric('frames_processed', float(len(frames)))
            
            return float(avg_score), float(max_score), len(frames)
        except Exception as e:
            logger.error(f"Error processing frame batch: {e}")
            self.metrics.system_healthy.set(0)
            return 0.0, 0.0, 0

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
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                    
                frames_batch.append(frame)
                frame_count += 1
                
                if len(frames_batch) >= batch_size:
                    avg_score, max_score, faces = self.process_frame_batch(frames_batch)
                    frames_data['scores'].append(float(avg_score))
                    frames_data['max_scores'].append(float(max_score))
                    frames_data['timestamps'].append(float(frame_count) / fps)
                    frames_data['faces_detected'].append(int(faces))
                    frames_batch = []
                    
                progress_bar.progress(min(frame_count / total_frames, 0.5))
                
            cap.release()
            
            if frames_batch:
                avg_score, max_score, faces = self.process_frame_batch(frames_batch)
                frames_data['scores'].append(float(avg_score))
                frames_data['max_scores'].append(float(max_score))
                frames_data['timestamps'].append(float(frame_count) / fps)
                frames_data['faces_detected'].append(int(faces))
            
            progress_bar.progress(0.7)
            
            # Start Spark streaming
            streaming_query = self.spark_processor.start_streaming(self.detector)
            
            # Transcribe
            transcription = self.whisper_model.transcribe(video_path)
            
            # Send to Kafka
            self.producer.send_video(video_path)
            
            final_score = float(np.mean(frames_data['max_scores']))
            return transcription["text"], final_score, frames_data
            
        except Exception as e:
            logger.error(f"Error analyzing video: {e}")
            self.metrics.system_healthy.set(0)
            raise

def create_monitoring_dashboard():
    st.subheader("System Monitoring")
    
    col1, col2, col3 = st.columns(3)
    
    try:
        system_metrics = metrics.get_system_metrics()
        
        with col1:
            cpu_usage = system_metrics['cpu_usage']
            st.metric("CPU Usage", f"{cpu_usage:.1f}%",
                     delta=f"{cpu_usage - 50:.1f}%" if cpu_usage > 50 else None,
                     delta_color="inverse")
            
        with col2:
            memory_usage = system_metrics['memory_usage']
            st.metric("Memory Usage", f"{memory_usage:.1f}%",
                     delta=f"{memory_usage - 70:.1f}%" if memory_usage > 70 else None,
                     delta_color="inverse")
            
        with col3:
            system_healthy = system_metrics['system_healthy']
            health_status = "Healthy" if system_healthy > 0.5 else "Unhealthy"
            health_delta = "OK" if system_healthy > 0.5 else "Check Logs"
            st.metric("System Health", health_status,
                     delta=health_delta,
                     delta_color="normal" if health_status == "Healthy" else "inverse")

        st.subheader("Performance Trends")
        
        # Create performance data
        now = datetime.now()
        times = [(now - timedelta(minutes=i)).strftime('%H:%M:%S') for i in range(10, -1, -1)]
        
        perf_data = pd.DataFrame({
            'Time': times,
            'CPU': [float(metrics.get_system_metrics()['cpu_usage']) for _ in times],
            'Memory': [float(metrics.get_system_metrics()['memory_usage']) for _ in times]
        })
        
        melted_data = pd.melt(
            perf_data,
            id_vars=['Time'],
            value_vars=['CPU', 'Memory'],
            var_name='Metric',
            value_name='Value'
        )
        melted_data['Value'] = melted_data['Value'].astype(float)
        
        perf_chart = alt.Chart(melted_data).mark_line().encode(
            x=alt.X('Time:T'),
            y=alt.Y('Value:Q', scale=alt.Scale(domain=[0, 100])),
            color='Metric:N',
            tooltip=['Time:T', 'Value:Q', 'Metric:N']
        ).properties(
            width=600,
            height=300
        )
        
        st.altair_chart(perf_chart, use_container_width=True)
        
    except Exception as e:
        logger.error(f"Error creating monitoring dashboard: {e}", exc_info=True)
        st.error("Unable to display monitoring dashboard")

def display_analysis_results(final_score: float, frames_data: Dict):
    col1, col2 = st.columns(2)
    
    with col1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=frames_data['timestamps'],
            y=frames_data['scores'],
            mode='lines',
            name='Average Score',
            line=dict(color='blue', width=2)
        ))
        fig.add_trace(go.Scatter(
            x=frames_data['timestamps'],
            y=frames_data['max_scores'],
            mode='lines',
            name='Max Score',
            line=dict(color='red', width=2)
        ))
        fig.update_layout(
            title='Deepfake Detection Confidence Over Time',
            xaxis_title='Time (seconds)',
            yaxis_title='Confidence Score',
            hovermode='x unified'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        hist_fig = go.Figure()
        hist_fig.add_trace(go.Histogram(
            x=frames_data['max_scores'],
            nbinsx=30,
            name='Score Distribution',
            marker_color='rgb(55, 83, 109)'
        ))
        hist_fig.update_layout(
            title='Detection Score Distribution',
            xaxis_title='Confidence Score',
            yaxis_title='Frequency',
            bargap=0.1
        )
        st.plotly_chart(hist_fig, use_container_width=True)
    
    st.subheader("Analysis Summary")
    summary_col1, summary_col2, summary_col3 = st.columns(3)
    
    with summary_col1:
        avg_score = float(np.mean(frames_data['max_scores']))
        st.metric(
            "Average Confidence",
            f"{avg_score:.2%}",
            delta=f"{avg_score - 0.5:.2%}",
            delta_color="inverse"
        )
    
    with summary_col2:
        max_score = float(max(frames_data['max_scores']))
        st.metric(
            "Peak Detection Score",
            f"{max_score:.2%}",
            delta=f"{max_score - avg_score:.2%}",
            delta_color="inverse"
        )
    
    with summary_col3:
        total_frames = len(frames_data['timestamps'])
        st.metric(
            "Total Frames Analyzed",
            f"{total_frames:,}",
            f"{total_frames/30:.1f} seconds"
        )
    
    if final_score > 0.7:
        st.error(f"🚨 High probability of deepfake detected (Confidence: {final_score:.2%})")
        st.markdown("""
            **Detection Details:**
            - Multiple frames show signs of manipulation
            - High confidence scores sustained over time
            - Consistent detection patterns across video segments
            
            **Recommended Actions:**
            - Conduct manual review
            - Check video metadata
            - Verify source authenticity
        """)
    elif final_score > 0.4:
        st.warning(f"⚠️ Potential manipulation detected (Confidence: {final_score:.2%})")
        st.markdown("""
            **Analysis Notes:**
            - Some suspicious frames detected
            - Moderate confidence in manipulation
            - Further investigation recommended
        """)
    else:
        st.success(f"✅ Video appears authentic (Confidence: {1-final_score:.2%})")
        st.markdown("""
            **Analysis Notes:**
            - No significant manipulation patterns detected
            - Low confidence scores across frames
            - Normal video characteristics observed
        """)
    
    with st.expander("Detailed Metrics"):
        metrics_df = pd.DataFrame({
            'Time (s)': frames_data['timestamps'],
            'Average Score': frames_data['scores'],
            'Max Score': frames_data['max_scores'],
            'Faces Detected': frames_data['faces_detected']
        })
        st.dataframe(
            metrics_df.style.background_gradient(subset=['Max Score'], cmap='RdYlGn_r'),
            use_container_width=True
        )
        
        csv = metrics_df.to_csv(index=False)
        st.download_button(
            label="Download Detailed Results",
            data=csv,
            file_name="deepfake_analysis_results.csv",
            mime="text/csv"
        )

def main():
    st.title("Advanced Video Analysis Pipeline with Monitoring")
    st.markdown("### Real-time Deepfake Detection & Transcription")
    
    # Add monitoring dashboard
    create_monitoring_dashboard()
    
    with st.sidebar:
        st.title("System Status")
        if st.button("Check Kafka Topics"):
            try:
                admin_client = KafkaAdminClient(bootstrap_servers=['localhost:29092'])
                topics = admin_client.list_topics()
                st.json({"Available Topics": topics})
            except Exception as e:
                st.error(f"Error checking topics: {e}")
                logger.error(f"Kafka error: {e}")
    
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
            
            try:

                analyzer = VideoAnalyzer()
                transcription, final_score, frames_data = analyzer.analyze_video(temp_path, progress_bar)
            except Exception as e:
                st.error(f"Error processing video: {str(e)}")
                logger.error(f"Processing error: {str(e)}", exc_info=True)
    # Add more specific error handling
                if "NoSuchMethodError" in str(e):
                    st.warning("Spark version compatibility issue detected. Please check system configurations.")
                return
            
            # Display analysis results
            display_analysis_results(final_score, frames_data)
            
            with st.expander("Video Transcription"):
                st.write(transcription)
                
            progress_bar.progress(1.0)
            
        except Exception as e:
            st.error(f"Error processing video: {str(e)}")
            logger.error(f"Processing error: {str(e)}")
            metrics.system_healthy.set(0)
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

if __name__ == "__main__":
    main()