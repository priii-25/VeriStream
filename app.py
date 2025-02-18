import cv2
import numpy as np
import time
import subprocess
import asyncio
import logging
from models import load_models  # Assuming this loads your models
from analyzer import OptimizedAnalyzer
from monitoring import MetricsCollector
import streamlink
import queue
import threading
import cProfile
import pstats
import streamlit as st
import os
from datetime import timedelta
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
from config import LANGUAGE_MAPPING
from logging_config import configure_logging
#from analyzer import OptimizedAnalyzer # Already imported
from video_analyzer import VideoAnalyzer
from utils import create_gis_map, create_monitoring_dashboard, display_analysis_results, visualize_knowledge_graph_interactive
from deep_translator import GoogleTranslator
from kafka.admin import KafkaAdminClient
from streamlit_folium import folium_static
import streamlit.components.v1 as components
#from realtime_stream_analyzer import RealtimeStreamAnalyzer # Will define below
import asyncio
import time
import cv2
import six
import sys
import plotly
import folium
import queue


# Ensure compatibility with Kafka client (Good practice, keep this)
if sys.version_info >= (3, 12, 0):
    sys.modules['kafka.vendor.six.moves'] = six.moves


logger = configure_logging()  # Assuming this sets up logging
metrics = MetricsCollector()  # Assuming this is for performance monitoring

class RealtimeStreamAnalyzer: # REPLACED with the class from the simplified script
    def __init__(self):
        self.is_running = False
        self.stream_process = None
        self.frame_queue = queue.Queue(maxsize=1000) # Increased queue size for buffering
        self.ffmpeg_thread = None
        self.frame_buffer = [] # Buffer to store frames

    def get_stream_resolution(self, stream_url):
        """Gets the resolution of a video stream using ffprobe."""
        try:
            command = [
                'ffprobe',
                '-v', 'error',
                '-select_streams', 'v:0',
                '-show_entries', 'stream=width,height',
                '-of', 'csv=s=x:p=0',
                stream_url
            ]
            result = subprocess.check_output(command).decode('utf-8').strip()
            parts = result.split('x')
            valid_parts = [part.strip() for part in parts if part.strip().isdigit()]

            if len(valid_parts) == 2:
                width, height = map(int, valid_parts)
                logger.info(f"Detected stream resolution: {width}x{height}")
                return width, height
            else:
                logger.warning(f"Unexpected ffprobe output: {result}. Using default resolution.")
                return 640, 480

        except Exception as e:
            logger.error(f"Error getting stream resolution: {e}", exc_info=True)
            return 640, 480

    async def _read_stream_ffmpeg(self, stream_url, buffer_duration=15):
        output_width = 320  # Or 160, as you prefer
        output_height = 180 # Or 90
        output_fps = 2      # Or even lower, like 1 or 0.5, if needed for stability
        frame_size = output_width * output_height * 3
        buffer_end_time = time.time() + buffer_duration

        command = [
        'ffmpeg',
        '-i', stream_url,
        '-vf', f'scale={output_width}:{output_height}', # Very low resolution scale
        '-r', str(output_fps),                         # EXTREMELY low FPS
        '-f', 'rawvideo',
        '-pix_fmt', 'rgb24',
        '-an',
        '-'
    ]

        try:
            self.stream_process = subprocess.Popen(
                command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=10**8
            )
            logger.info(f"FFmpeg process started with PID: {self.stream_process.pid}")

            def log_ffmpeg_stderr(stderr_pipe):
                while self.is_running and self.stream_process:
                    if stderr_pipe.closed: # <--- Add this check
                        break
                    line = stderr_pipe.readline().decode('utf-8', errors='ignore').strip()
                    if line:
                        logger.error(f"FFmpeg Stderr: {line}")
                    elif self.stream_process.poll() is not None:
                        break

            stderr_thread = threading.Thread(target=log_ffmpeg_stderr, args=(self.stream_process.stderr,), daemon=True)
            stderr_thread.start()

            self.frame_buffer = [] # Clear buffer before starting

            while self.is_running and self.stream_process and time.time() < buffer_end_time: # Buffer for duration
                try:
                    start_read_time = time.time() # <--- Add start time
                    raw_frame = self.stream_process.stdout.read(frame_size)
                    read_duration = time.time() - start_read_time # <--- Calculate read duration

                    logger.info(f"Frame queue size: {self.frame_queue.qsize()}, Frame buffer size: {len(self.frame_buffer)}, FFmpeg read time: {read_duration:.4f}s") # <--- ADD THIS LOGGING


                    if not raw_frame:
                        if self.stream_process.poll() is not None:
                            logger.warning("No frame data and FFmpeg process ended.")
                            break
                        else:
                            continue

                    if len(raw_frame) != frame_size:
                        continue

                    frame = np.frombuffer(raw_frame, dtype=np.uint8).reshape((output_height, output_width, 3))
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    if frame.size == 0:
                        continue
                    if np.any(np.isnan(frame)):
                        continue

                    self.frame_buffer.append(frame) # Buffer frame

                except Exception as e:
                    logger.error(f"Error reading frame: {e}", exc_info=True)
                    break

        except Exception as e:
            logger.error(f"FFmpeg process encountered an error: {e}")

        finally:
            if self.stream_process:
                self.stream_process.stdout.close()
                if self.stream_process.stderr:
                    self.stream_process.stderr.close()
                logger.info("FFmpeg process cleaned up.")
            self.is_running = False # Stop after buffering is done or error occurs


    async def start_analysis(self, stream_url):
        """Start buffering."""
        self.is_running = True
        self.frame_buffer = [] # Ensure buffer is clear at start

        try:
            streams = streamlink.streams(stream_url)
            if not streams:
                raise ValueError(f"No streams found for URL: {stream_url}")

            stream = streams.get("720p")
            if stream is None:
                stream = streams.get("480p")
            if stream is None:
                stream = streams.get("360p")
            if stream is None:
                stream = streams.get("best")

            actual_stream_url = stream.url
            logger.info(f"Resolved stream URL: {actual_stream_url}")

            self.ffmpeg_thread = asyncio.create_task(self._read_stream_ffmpeg(actual_stream_url))

        except streamlink.StreamlinkError as e:
            logger.error(f"Streamlink error: {e}")
            st.error(f"Streamlink error: {e}")
            self.is_running = False
        except ValueError as e:
            logger.error(f"Value error: {e}")
            st.error(f"Value error: {e}")
            self.is_running = False
        except Exception as e:
            logger.error(f"An unexpected error occurred: {e}")
            st.error(f"An unexpected error occurred: {e}")
            self.is_running = False


    def stop_analysis(self):
        """Stop analysis and clean up."""
        self.is_running = False

        if self.stream_process:
            try:
                self.stream_process.send_signal(subprocess.signal.SIGINT)
                self.stream_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                logger.warning("FFmpeg process did not terminate gracefully.  Forcing termination.")
                self.stream_process.kill()
            finally:
                self.stream_process = None

        if self.ffmpeg_thread and not self.ffmpeg_thread.done():
             self.ffmpeg_thread.cancel()
             try:
                asyncio.run(self.ffmpeg_thread)
             except asyncio.CancelledError:
                pass

        self.frame_buffer = [] # Clear buffer on stop
        logger.info("Real-time analysis stopped.")


    def start_profiling(self): # Keep this, but its functionality is now different in real-time page
        """Start profiling."""
        self.profiling = True
        print("Profiling started. Run analysis, then stop to see results.")

    def stop_profiling(self): # Keep this, but its functionality is now different in real-time page
        """Stop profiling and print results."""
        self.profiling = False
        if self.profiler:
             self.profiler.disable()
             stats = pstats.Stats(self.profiler)
             stats.sort_stats(pstats.SortKey.TIME)
             stats.print_stats(20)


# app.py (Corrected for Streamlit Display)

class OutbreakAnalyzer:
    def __init__(self, data_path):
        self.df = pd.read_csv(data_path)
        self.df['date'] = pd.to_datetime(self.df['date'])

    def visualize_trends(self, include_predictions=True, prediction_days=30):
        try:
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=(
                    'Outbreaks Over Time',
                    'Outbreaks by Region',
                    'Outbreaks by Reason',
                    'Region-Reason Heatmap'
                )
            )

            df_time = self.df.groupby('date').size().reset_index(name='count')
            fig.add_trace(
                go.Scatter(x=df_time['date'], y=df_time['count'], name='Historical Outbreaks'),
                row=1, col=1
            )

            if include_predictions:
                end_date = self.df['date'].max() + timedelta(days=prediction_days)
                predictions = predict_range(
                    self.df['date'].max() + timedelta(days=1),
                    end_date
                )
                if predictions is not None and len(predictions) > 0:
                    pred_time = predictions.groupby('date').size().reset_index(name='count')
                    fig.add_trace(
                        go.Scatter(x=pred_time['date'], y=pred_time['count'],
                                 name='Predicted Outbreaks', line=dict(dash='dash')),
                        row=1, col=1
                    )

            region_counts = self.df['region'].value_counts()
            fig.add_trace(
                go.Bar(x=region_counts.index, y=region_counts.values, name='Outbreaks by Region'),
                row=1, col=2
            )

            reason_counts = self.df['reason'].value_counts()
            fig.add_trace(
                go.Bar(x=reason_counts.index, y=reason_counts.values, name='Outbreaks by Reason'),
                row=2, col=1
            )

            heatmap_data = pd.crosstab(self.df['region'], self.df['reason'])
            fig.add_trace(
                go.Heatmap(z=heatmap_data.values, x=heatmap_data.columns,
                          y=heatmap_data.index, name='Region-Reason Distribution'),
                row=2, col=2
            )

            fig.update_layout(
                height=800,
                width=1200,
                title_text="Outbreak Analysis Dashboard",
                showlegend=True
            )

            return fig
        except Exception as e:
            print(f"Error creating visualization: {str(e)}")
            return None

def predict_range(start_date, end_date, model_path='outbreak_model.joblib'):
    try:
        model_components = joblib.load(model_path)
        model = model_components['model']
        le_region = model_components['le_region']
        le_reason = model_components['le_reason']
        feature_columns = model_components['feature_columns']
        regions = model_components['regions']
        reasons = model_components['reasons']

        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)
        dates = pd.date_range(start=start_date, end=end_date)

        prediction_data = []
        for date in dates:
            for region in regions:
                for reason in reasons:
                    prediction_data.append({
                        'date': date,
                        'region': region,
                        'reason': reason,
                        'year': date.year,
                        'month': date.month,
                        'day': date.day,
                        'day_of_week': date.dayofweek,
                        'region_encoded': le_region.transform([region])[0],
                        'reason_encoded': le_reason.transform([reason])[0],
                        'region_frequency': 1,
                        'reason_frequency': 1,
                        'region_7d_count': 0,
                        'region_30d_count': 0,
                        'region_90d_count': 0,
                        'reason_7d_count': 0,
                        'reason_30d_count': 0,
                        'reason_90d_count': 0
                    })

        prediction_df = pd.DataFrame(prediction_data)
        predictions = model.predict_proba(prediction_df[feature_columns])
        prediction_df['outbreak_probability'] = predictions.max(axis=1)
        high_risk_outbreaks = prediction_df[prediction_df['outbreak_probability'] > 0.8]

        return high_risk_outbreaks[['date', 'region', 'reason', 'outbreak_probability']]
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        return None

def video_analysis_page():
    st.title("VERISTREAM")
    st.markdown("### Real-time Deepfake Detection & Transcription")

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

            analyzer = VideoAnalyzer()
            transcription, final_score, frames_data = analyzer.analyze_video(temp_path, progress_bar)

            text_analyzer = OptimizedAnalyzer(use_gpu=True)
            analysis_result = text_analyzer.analyze_text(transcription)

            display_analysis_results(final_score, frames_data)

            with st.expander("Video Transcription"):
                st.write(transcription)
                target_language = st.selectbox(
                    "Translate to",
                    ["English", "Assamese", "Bengali", "Gujarati", "Hindi", "Kannada",
                     "Malayalam", "Marathi", "Odia (Oriya)", "Urdu"]
                )

                if target_language != "English":
                    target_code = LANGUAGE_MAPPING.get(target_language, "en")
                    translated_text = GoogleTranslator(source='auto', target=target_code).translate(transcription)
                    st.write(f"Translated to {target_language}:")
                    st.write(translated_text)

            with st.expander("Text Analysis Results"):
                st.write("### Sentiment Analysis")
                if analysis_result.sentiment:
                    st.write(f"**Sentiment:** {analysis_result.sentiment['label']}")
                    st.write(f"**Confidence Score:** {analysis_result.sentiment['score']:.4f}")
                else:
                    st.write("Sentiment analysis not available")

                for section in ["Fact Checks", "Emotional Triggers", "Stereotypes",
                              "Manipulation Score", "Entities", "Generative Analysis"]:
                    st.write(f"### {section}")
                    st.write(getattr(analysis_result, section.lower().replace(" ", "_")))

                st.write("### Knowledge Graph")
                visualize_knowledge_graph_interactive(analysis_result.knowledge_graph)

                st.subheader("Geospatial Visualization of Detected Locations")
                gis_map = create_gis_map()
                folium_static(gis_map)

            progress_bar.progress(1.0)

        except Exception as e:
            st.error(f"Error processing video: {str(e)}")
            logger.error(f"Processing error: {str(e)}", exc_info=True)
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

def analytics_prediction_page():
    st.title("Analytics & Predictions")

    try:
        analyzer = OutbreakAnalyzer('misinformation_dataset.csv')

        # Create two tabs
        tab1, tab2 = st.tabs(["Trend Analysis & Predictions", "Misinformation Spread"])

        with tab1:
            st.subheader("Prediction Settings")
            prediction_days = st.slider("Number of days to predict", 7, 90, 30)
            include_predictions = st.checkbox("Include predictions", value=True)

            # Visualization section
            fig = analyzer.visualize_trends(
                include_predictions=include_predictions,
                prediction_days=prediction_days
            )

            if fig:
                st.plotly_chart(fig, use_container_width=True)

                # Show predictions table in the same tab
                if include_predictions:
                    st.subheader("Detailed Predictions")
                    predictions = predict_range(
                        start_date=pd.Timestamp.now(),
                        end_date=pd.Timestamp.now() + timedelta(days=prediction_days)
                    )

                    if predictions is not None:
                        col1, col2 = st.columns([2, 1])
                        with col1:
                            st.dataframe(
                                predictions.sort_values('outbreak_probability', ascending=False),
                                use_container_width=True
                            )
                        with col2:
                            st.metric(
                                "Total High-Risk Outbreaks",
                                len(predictions),
                                help="Predicted outbreaks with probability > 0.8"
                            )
            else:
                st.error("Error generating visualizations")

        with tab2:
            st.subheader("Misinformation Spread Visualization")
            try:
                # Read the HTML file
                html_file_path = "misinformation_map.html"
                if os.path.exists(html_file_path):
                    with open(html_file_path, 'r', encoding='utf-8') as f:
                        html_content = f.read()
                    # Render the HTML content in full width
                    components.html(html_content, height=800, scrolling=True)
                else:
                    st.error("HTML file not found. Please ensure html exists in the application directory.")
            except Exception as e:
                st.error(f"Error loading HTML visualization: {str(e)}")
                logger.error(f"HTML rendering error: {str(e)}", exc_info=True)

    except Exception as e:
        st.error(f"Error in analytics page: {str(e)}")
        logger.error(f"Analytics error: {str(e)}", exc_info=True)

def realtime_analysis_page(): # UPDATED for buffering and display
    st.title("Real-time Stream Analysis") # Or "Real-time Stream Display"

    stream_url = st.text_input("Enter Stream URL (YouTube, Twitch, etc.):", key="stream_url")

    col1, col2, col3 = st.columns(3)
    with col1:
        start_button = st.button("Start Stream (10 seconds)") 
    with col2:
        stop_button = st.button("Stop Stream")
    with col3:
        profile_button = st.button("Profile Analysis") 

    if 'analyzer' not in st.session_state:
        st.session_state.analyzer = RealtimeStreamAnalyzer()

    if 'display_frame' not in st.session_state:
        st.session_state.display_frame = None

    frame_container = st.empty()

    def start_streaming_task(url):
        try:
        
            async def buffer_and_display():
                await st.session_state.analyzer.start_analysis(url)
                buffer_wait_time = 0
                while len(st.session_state.analyzer.frame_buffer) < 10 and buffer_wait_time < 10:
                    await asyncio.sleep(1)
                    buffer_wait_time += 1
                if len(st.session_state.analyzer.frame_buffer) > 0:
                    frame_index = 0
                    while frame_index < len(st.session_state.analyzer.frame_buffer):
                        if not st.session_state.analyzer.is_running:
                            break
                    
                        frame = st.session_state.analyzer.frame_buffer[frame_index]
                        if frame is not None and frame.size > 0:
                            frame_container.image(frame, channels="RGB", use_column_width=True)
                            st.write(f"Displaying frame {frame_index + 1}/{len(st.session_state.analyzer.frame_buffer)}")
                    
                        frame_index += 1
                        await asyncio.sleep(0.1) 
                
                    st.write(f"Finished displaying {frame_index} frames")
                else:
                    st.write("No frames were buffered!")

            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(buffer_and_display())
        
        except Exception as e:
            st.error(f"Error in streaming task: {str(e)}")
            logger.error(f"Streaming error: {str(e)}", exc_info=True)

    async def display_buffered_stream(duration=10):
        if not st.session_state.analyzer.frame_buffer:
            st.warning("No frames buffered yet. Start streaming first.")
            return

        frame_index = 0
        total_frames = len(st.session_state.analyzer.frame_buffer)
        start_time = time.time()
    
        progress = st.progress(0)
        frames_displayed = 0
    
        while (time.time() - start_time) < duration and frame_index < total_frames:
            try:
                frame = st.session_state.analyzer.frame_buffer[frame_index]
            
                if frame is not None and frame.size > 0:
                    frame_container.image(frame, channels="RGB", use_column_width=True)
                    frames_displayed += 1
                    progress.progress(frame_index / total_frames)
                    if frames_displayed % 10 == 0:  
                        logger.info(f"Displayed {frames_displayed}/{total_frames} frames")
                    frame_index += 1
                    await asyncio.sleep(0.1)
                else:
                    logger.warning(f"Skipping invalid frame at index {frame_index}")
                    frame_index += 1
                
            except IndexError:
                logger.error("Reached end of frame buffer")
                break
            except Exception as e:
                logger.error(f"Error displaying frame: {e}")
                break
        
            if not st.session_state.analyzer.is_running:
                logger.info("Display stopped: analyzer not running")
                break
    
        progress.progress(1.0)
        elapsed_time = time.time() - start_time
        st.write(f"Display finished: {frames_displayed} frames shown in {elapsed_time:.2f} seconds")
        st.write(f"Effective frame rate: {frames_displayed/elapsed_time:.2f} FPS")
        if frame_index >= total_frames:
            st.session_state.analyzer.frame_buffer = []  
        st.write("Stream display finished.")

    if start_button:
        if st.session_state.analyzer.is_running:
            st.warning("Stream is already running/buffering.")
        else:
            start_streaming_task(stream_url) # Start buffering
            time.sleep(5) # Give buffer some time to fill (adjust if needed)
            asyncio.run(display_buffered_stream(duration=10)) # Display for 10 seconds, matching button text

    if stop_button:
        if st.session_state.analyzer:
            st.session_state.analyzer.stop_analysis()
            frame_container.empty()

    if profile_button:  # Profile button functionality in real-time page is now limited to buffering
        st.warning("Profiling in Real-time Stream page will only profile the buffering stage.")
        if st.session_state.analyzer and st.session_state.analyzer.is_running:
            st.warning("Stop the current analysis before profiling.")
        else:
          st.session_state.analyzer.start_profiling()
          start_streaming_task(stream_url) # Start buffering with profiling
          st.session_state.analyzer.stop_analysis() # Stop after buffering (and profiling) is done
          st.session_state.analyzer.stop_profiling() # Stop profiling and print results


def main():
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Video Analysis", "Analytics & Predictions", "Real-time Analysis"])

    if page == "Video Analysis":
        video_analysis_page()
    elif page == "Analytics & Predictions":
        analytics_prediction_page()
    elif page == "Real-time Analysis":
        realtime_analysis_page()

if __name__ == "__main__":
    main()