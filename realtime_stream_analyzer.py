import cv2
import numpy as np
import time
import subprocess
import asyncio
import logging
from models import load_models
from analyzer import OptimizedAnalyzer
from monitoring import MetricsCollector
import streamlink
import queue
import threading
import cProfile
import pstats

logger = logging.getLogger('veristream_realtime')
metrics = MetricsCollector()

class RealtimeStreamAnalyzer:
    def __init__(self):
        self.whisper_model, self.detector = load_models()
        self.text_analyzer = OptimizedAnalyzer(use_gpu=True)
        self.is_running = False
        self.stream_process = None
        self.frame_count = 0
        self.last_frame_time = None
        self.last_deepfake_check = time.time()
        self.frame_queue = queue.Queue(maxsize=2000)
        self.last_score_update_time = time.time()
        self.score_update_interval = 20
        self.result_queue = queue.Queue()
        self.ffmpeg_thread = None
        self.profiler = cProfile.Profile()  # Create a profiler instance
        self.profiling = False

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

    async def process_frame_batch(self, frames):
        pass

    async def frame_processing_task(self):
        batch_size = 16  # Process in batches
        frames_buffer = []
        last_detection_time = time.time()
        detection_interval = 2.0  # Seconds between detections

        logger.info("Starting frame_processing_task")
        while self.is_running:
            try:
                logger.info("frame_processing_task: Trying to get frame from frame_queue")
                frame = self.frame_queue.get(timeout=1)
                logger.info("frame_processing_task: Frame successfully retrieved from frame_queue")

                frames_buffer.append(frame)  # Accumulate frames

                if len(frames_buffer) >= batch_size or (time.time() - last_detection_time) >= detection_interval:
                # Process the batch
                    if frames_buffer: # Check if the buffer isn't empty
                        processed_frames = [self.process_frame(f)['frame'] for f in frames_buffer] # process frames
                        scores = self.detector.predict_batch(processed_frames) # get the scores
                        avg_score = sum(scores) / len(scores) if scores else 0.0 # average the scores
                    # Send to Kafka Here
                        self.producer.send_frame(processed_frames[0])
                    # Update result queue
                        self.result_queue.put({
                            'frame': processed_frames[-1],  # Display the *last* frame of the batch
                            'score': avg_score,
                            'processing_time': time.time() - last_detection_time #rough estimate
                        })

                        last_detection_time = time.time()
                        frames_buffer = []  # Clear the buffer

            except queue.Empty:
                logger.info("frame_processing_task: frame_queue is empty")
            except Exception as e:
                logger.error(f"Error in frame_processing_task: {e}", exc_info=True)
                await asyncio.sleep(1)  # Prevent busy-loop on error

        logger.info("frame_processing_task: Exiting task loop")

    async def _read_stream_ffmpeg(self, stream_url):
        """Read stream, put frames into frame_queue."""

        output_width = 320  # Reduced resolution
        output_height = 180 # Reduced resolution
        output_fps = 5      # Reduced frame rate

        width, height = self.get_stream_resolution(stream_url)
        frame_size = output_width * output_height * 3

        command = [
            'ffmpeg',
            '-i', stream_url,
            '-vf', f'scale={output_width}:{output_height}',
            '-r', str(output_fps),
            '-f', 'rawvideo',
            '-pix_fmt', 'bgr24',
            '-an',
            '-'
        ]

        try:
            self.stream_process = subprocess.Popen(
                command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=10**8
            )
            logger.info(f"FFmpeg process started with PID: {self.stream_process.pid} "
                        f"with resolution {output_width}x{output_height} and {output_fps} FPS.")

            def log_ffmpeg_stderr(stderr_pipe):
                while self.is_running and self.stream_process:
                    line = stderr_pipe.readline().decode('utf-8', errors='ignore').strip()
                    if line:
                        logger.error(f"FFmpeg Stderr: {line}")
                    elif self.stream_process.poll() is not None:
                        break

            stderr_thread = threading.Thread(target=log_ffmpeg_stderr, args=(self.stream_process.stderr,), daemon=True)
            stderr_thread.start()


            while self.is_running and self.stream_process:
                try:
                    logger.info("_read_stream_ffmpeg: Attempting to read frame data from pipe")
                    raw_frame = self.stream_process.stdout.read(frame_size)
                    logger.info(f"_read_stream_ffmpeg: Read {len(raw_frame)} bytes of frame data")
                    if not raw_frame:
                        if self.stream_process.poll() is not None:
                            logger.warning("No frame data and FFmpeg process ended.")
                            break
                        else:
                            continue

                    frame = np.frombuffer(raw_frame, dtype=np.uint8).reshape((output_height, output_width, 3))

                    try:
                        self.frame_queue.put_nowait(frame)
                    except queue.Full:
                        logger.warning("Frame queue is full. Dropping frame for display.")

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
                # No need to kill here; stop_analysis handles it
                logger.info("FFmpeg process cleaned up.")

    async def start_analysis(self, stream_url):
        """Start the analysis."""
        import streamlit as st
        self.is_running = True
        self.frame_count = 0
        self.last_deepfake_check = time.time()

        if self.profiling:
            self.profiler.enable()  # Start profiling

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
            asyncio.create_task(self.frame_processing_task())

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


    def process_frame(self, frame):
        """Process frame."""
        try:
            start_time = time.time()

            if frame is None:
                logger.error("Received None frame")
                return None

            processed_frame = frame.copy()
            processed_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
            logger.info(f"Frame shape: {processed_frame.shape}, dtype: {processed_frame.dtype}")

            processing_time = time.time() - start_time

            return {
                'frame': processed_frame,
                'score': 0.0,
                'processing_time': processing_time
            }

        except Exception as e:
            logger.error(f"Frame processing error: {e}")
            return None
    def stop_analysis(self):
        """Stop analysis and clean up."""
        self.is_running = False  # Signal threads to stop

        if self.profiling:
            self.profiler.disable()
            stats = pstats.Stats(self.profiler)
            stats.sort_stats(pstats.SortKey.TIME)
            stats.print_stats(20)
            self.profiling = False

        # 1. Signal FFmpeg to stop *before* terminating: Send SIGINT
        if self.stream_process:
            try:
                self.stream_process.send_signal(subprocess.signal.SIGINT)  # Graceful stop
                self.stream_process.wait(timeout=5)  # Wait for it to finish
            except subprocess.TimeoutExpired:
                logger.warning("FFmpeg process did not terminate gracefully.  Forcing termination.")
                self.stream_process.kill()  # Forceful termination if needed
            finally:
                self.stream_process = None

        # 2.  Wait for the FFmpeg thread to finish (if you used one)
        if self.ffmpeg_thread and not self.ffmpeg_thread.done():
             self.ffmpeg_thread.cancel() #Cancel the task
             try:
                asyncio.run(self.ffmpeg_thread) # Wait to finish
             except asyncio.CancelledError:
                pass

        # 3. Clear the queues
        while not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
            except queue.Empty:
                break
        while not self.result_queue.empty():
            try:
                self.result_queue.get_nowait()
            except queue.Empty:
                break

        logger.info("Real-time analysis stopped.")


    def start_profiling(self):
        """Start profiling."""
        self.profiling = True
        print("Profiling started. Run analysis, then stop to see results.")

    def stop_profiling(self):
        """Stop profiling and print results."""
        self.profiling = False  # Set profiling to False *before* disabling
        if self.profiler:
             self.profiler.disable()
             stats = pstats.Stats(self.profiler)
             stats.sort_stats(pstats.SortKey.TIME)
             stats.print_stats(20)