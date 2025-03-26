import cv2
import numpy as np
import subprocess
import asyncio
import logging
from models import load_models
from analyzer import OptimizedAnalyzer
from video_processor import VideoProducer
from spark_video_processor import SparkVideoProcessor
from utils import display_realtime_results
import streamlit as st
import time
import gi
logger = logging.getLogger('veristream')

class StreamAnalyzer:
    def __init__(self):
        self.whisper_model, self.detector = load_models()
        self.text_analyzer = OptimizedAnalyzer(use_gpu=True)
        self.spark_processor = SparkVideoProcessor()
        self.video_producer = VideoProducer()
        self.is_running = False
        self.stream_process = None

    async def start_rtmp_stream(self, stream_url):
        """Start RTMP stream using FFmpeg and process frames"""
        command = [
            'ffmpeg',
            '-i', stream_url,
            '-f', 'image2pipe',
            '-pix_fmt', 'bgr24',
            '-vcodec', 'rawvideo',
            '-'
        ]
        
        self.stream_process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=10**8
        )

        while self.is_running:
            try:
                raw_image = self.stream_process.stdout.read(480*640*3)
                if len(raw_image) == 0:
                    break
                frame = np.frombuffer(raw_image, dtype=np.uint8).reshape(480, 640, 3)
                await self.process_frame(frame)
            except Exception as e:
                logger.error(f"RTMP processing error: {e}")
                break

    async def start_webrtc_stream(self, webrtc_url):
        """Start WebRTC stream using GStreamer"""
        pipeline = f"""
            rtspsrc location={webrtc_url} !
            application/x-rtp,media=video,encoding-name=H264 !
            rtph264depay !
            avdec_h264 !
            videoconvert !
            appsink
        """
        gi.require_version('Gst', '1.0')
        from gi.repository import Gst

        Gst.init(None)
        loop = asyncio.get_event_loop()
        bus = self.pipeline.get_bus()
        
        def on_message(bus, message):
            if message.type == Gst.MessageType.EOS:
                loop.stop()
        
        bus.add_signal_watch()
        bus.connect("message", on_message)
        
        while self.is_running:
            try:
                sample = self.appsink.pull_sample()
                frame = sample.get_buffer().extract_dup(0, sample.get_size())
                frame = np.frombuffer(frame, dtype=np.uint8)
                await self.process_frame(frame)
            except Exception as e:
                logger.error(f"WebRTC processing error: {e}")
                break

    async def process_frame(self, frame):
        """Common frame processing for both streams"""
        try:
            start_time = time.time()
            avg_score, max_score = self.detector.predict_batch([frame])
            processing_time = time.time() - start_time
            self.spark_processor.process_frame(frame)
            self.metrics.record_metric('processing_time', processing_time)
            self.metrics.record_metric('frames_processed', 1)
            st.session_state.latest_frame = frame
            st.session_state.analysis_metrics = {
                'latest_score': max_score,
                'processing_time': processing_time
            }
        except Exception as e:
            logger.error(f"Frame processing error: {e}")

    def start_analysis(self, stream_url):
        """Start analysis based on stream type"""
        self.is_running = True
        if stream_url.startswith('rtmp'):
            asyncio.create_task(self.start_rtmp_stream(stream_url))
        elif stream_url.startswith('webrtc'):
            asyncio.create_task(self.start_webrtc_stream(stream_url))
        else:
            raise ValueError("Unsupported stream type")

    def stop_analysis(self):
        """Stop all processing"""
        self.is_running = False
        if self.stream_process:
            self.stream_process.terminate()
            self.stream_process = None