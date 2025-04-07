# backend/main.py
from fastapi import FastAPI, UploadFile, File, WebSocket, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
# --- Keep imports needed for BOTH modes ---
from video_analyzer import VideoAnalyzer # Used by both
from analyzer import OptimizedAnalyzer # Used by both (or parts of it)
from optimized_deepfake_detector import OptimizedDeepfakeDetector # Used directly by stream AND by video_analyzer
from knowledge_graph import KnowledgeGraphManager # Used by Analyzer/FactChecker
from fact_checker import FactChecker # Used by both
import cv2
import numpy as np
import asyncio
import time
import logging
import subprocess
import os
import json
import requests
from dotenv import load_dotenv
from typing import Dict, Any, List, Optional # Added Optional
import aiofiles
import aiofiles.os
from streamlink import Streamlink

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

HISTORY_LIMIT = 15
CLEANUP_BUFFER = 5

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, # Set default to INFO, DEBUG can be too verbose
    format='%(asctime)s - %(levelname)s - [%(threadName)s] - %(message)s',
    handlers=[logging.StreamHandler(), logging.FileHandler('veristream_backend.log')]
)
# Silence overly verbose libraries unless debugging
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING) # Added httpcore
logging.getLogger("shap").setLevel(logging.WARNING)
logging.getLogger("kafka").setLevel(logging.WARNING)
logging.getLogger("PIL").setLevel(logging.WARNING) # Pillow logs font warnings sometimes
logging.getLogger("matplotlib").setLevel(logging.WARNING) # Matplotlib logs


# Suppress Hugging Face parallelism warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Ensure temp directory exists
TEMP_DIR = "temp_media"
if not os.path.exists(TEMP_DIR):
    os.makedirs(TEMP_DIR)

app = FastAPI()
app.mount(f"/{TEMP_DIR}", StaticFiles(directory=TEMP_DIR), name="temp_files")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"], # Adjust as needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Instantiate components used by BOTH modes ---
logger.info("Initializing models and components...")
try:
    # The single detector instance used by the stream
    stream_deepfake_detector = OptimizedDeepfakeDetector()
    # VideoAnalyzer now uses its OWN detector instance internally, no need to pass one
    video_analyzer = VideoAnalyzer()
    text_analyzer = OptimizedAnalyzer(use_gpu=True) # Or False if no GPU
    fact_checker = FactChecker()
    streamlink_session = Streamlink()
    # Instantiate Kafka producer if needed for upload Option 1
    video_producer = None # Initialize as None
    # try:
    #     video_producer = VideoProducer() # Uncomment if using Kafka uploads
    # except Exception as kafka_err:
    #     logger.error(f"Failed to initialize Kafka Producer: {kafka_err}. Upload via Kafka will fail.")

    logger.info("Models and components initialized.")
except Exception as e:
     logger.critical(f"FATAL: Failed to initialize core components: {e}", exc_info=True)
     # Exit or raise a more specific error if core components fail
     raise RuntimeError(f"Core component initialization failed: {e}") from e

# --- Global State for STREAM mode ---
stream_is_running = False
stream_result_queue = asyncio.Queue()
stream_fact_check_buffer = ""
STREAM_FACT_CHECK_BUFFER_DURATION = 30
stream_last_fact_check_time = 0
stream_active_tasks = set()
# Define heatmap threshold for stream mode
STREAM_HEATMAP_THRESHOLD = 0.7

# --- Global State for UPLOAD mode ---
upload_progress_clients = []


# === STREAM MODE FUNCTIONS ===

async def stream_download_chunk(stream_url: str, output_path: str, duration: int) -> bool:
    """Downloads AND RE-ENCODES a stream chunk using FFmpeg."""
    task_id = asyncio.current_task().get_name() if asyncio.current_task() else 'download'
    logger.info(f"[{task_id}] Starting download & re-encode: {os.path.basename(output_path)} for {duration}s")
    # Reduced CRF for potentially better quality if needed, but slower
    cmd = [ "ffmpeg", "-hide_banner", "-loglevel", "warning",
        "-i", stream_url, "-t", str(duration),
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "26", "-pix_fmt", "yuv420p", # Adjusted preset/crf
        "-c:a", "aac", "-b:a", "128k", "-f", "mp4", output_path, "-y" ]
    process = await asyncio.create_subprocess_exec(
        *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE )
    stdout, stderr = await process.communicate()
    if process.returncode != 0:
        logger.error(f"[{task_id}] FFmpeg download failed for {os.path.basename(output_path)} (Code: {process.returncode}): {stderr.decode()}")
        return False
    else:
        # Use aiofiles for file system checks
        try:
            if await aiofiles.os.path.exists(output_path):
                stat_result = await aiofiles.os.stat(output_path)
                if stat_result.st_size > 1000: # Check > 1KB approx
                     logger.info(f"[{task_id}] Finished download (re-encoded): {os.path.basename(output_path)}")
                     return True
                else:
                     logger.error(f"[{task_id}] FFmpeg finished but output file {os.path.basename(output_path)} is too small.")
                     await aiofiles.os.remove(output_path) # Clean up tiny file
                     return False
            else:
                logger.error(f"[{task_id}] FFmpeg finished but output file {os.path.basename(output_path)} is missing.")
                return False
        except Exception as file_err:
             logger.error(f"[{task_id}] Error checking/removing output file {os.path.basename(output_path)}: {file_err}")
             return False

async def stream_analyze_chunk(chunk_path: str, chunk_index: int) -> Dict[str, Any] | None:
    """Analyzes a single video chunk (stream mode), generating heatmap if score is high."""
    task_id = asyncio.current_task().get_name() if asyncio.current_task() else f'analyze-{chunk_index}'
    if not await aiofiles.os.path.exists(chunk_path):
        logger.warning(f"[{task_id}] Chunk file not found: {chunk_path}")
        return None

    logger.info(f"[{task_id}] Analyzing chunk: {os.path.basename(chunk_path)}")
    analysis_start_time = time.time()
    analysis_data = {
        "deepfake_analysis": {"timestamp": -1.0, "score": 0.0, "heatmap_url": None}, # Initialize heatmap_url
        "transcription": {"original": "[Analysis Pending]", "detected_language": "unknown", "english": ""}
    }
    analysis_frame = None # The single frame chosen for analysis
    actual_frame_time = -1.0
    cap = None
    audio_path = os.path.join(TEMP_DIR, f"stream_audio_{chunk_index}.wav")
    overlay_path_abs = None # Store absolute path for cleanup if generated
    base_chunk_filename = os.path.basename(chunk_path).split('.')[0] # e.g., stream_chunk_123

    try:
        # --- Video Analysis ---
        cap = cv2.VideoCapture(chunk_path)
        if not cap.isOpened(): raise ValueError(f"Failed to open chunk {os.path.basename(chunk_path)}")

        # Target the middle frame (e.g., 5 seconds into a 10-second chunk)
        target_frame_time_ms = 5 * 1000
        frame_read_count = 0
        fps = cap.get(cv2.CAP_PROP_FPS) or 30 # Default FPS if not available
        target_frame_index = int(fps * 5) # Approx frame index for 5 seconds

        while True:
            ret, frame = cap.read()
            if not ret: break
            current_time_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
            # Select the frame closest to the target time
            if analysis_frame is None and frame_read_count >= target_frame_index:
                if frame is not None and frame.size > 0:
                    analysis_frame = frame.copy()
                    actual_frame_time = current_time_ms / 1000.0
                    logger.debug(f"[{task_id}] Selected frame at index {frame_read_count} (~{actual_frame_time:.2f}s)")
                    break # Found our frame
                else:
                    logger.warning(f"[{task_id}] Invalid frame near target index {target_frame_index}, continuing search...")
            frame_read_count += 1
        # Ensure cap is released even if loop breaks early
        if cap.isOpened(): cap.release()

        # --- Deepfake Detection & Heatmap Generation ---
        deepfake_score = 0.0
        heatmap_url = None
        if analysis_frame is not None:
            try:
                # Use predict_batch (even for single frame, keeps API consistent)
                # Expects list of frames
                scores = await asyncio.to_thread(stream_deepfake_detector.predict_batch, [analysis_frame])
                deepfake_score = scores[0] if scores else 0.0
                logger.debug(f"[{task_id}] Deepfake score: {deepfake_score:.3f}")

                # Generate heatmap ONLY if score is high
                if deepfake_score > STREAM_HEATMAP_THRESHOLD:
                    logger.info(f"[{task_id}] Score {deepfake_score:.3f} > {STREAM_HEATMAP_THRESHOLD}, generating heatmap...")
                    # These are blocking calls, run in thread
                    heatmap = await asyncio.to_thread(stream_deepfake_detector.get_attention_maps, analysis_frame)
                    if heatmap is not None:
                        overlay = await asyncio.to_thread(stream_deepfake_detector.create_overlay, analysis_frame, heatmap)
                        if overlay is not None:
                            overlay_filename = f"{base_chunk_filename}_overlay.jpg"
                            overlay_path_abs = os.path.join(TEMP_DIR, overlay_filename)
                            # Save using OpenCV in the thread
                            def save_overlay():
                                try:
                                    success = cv2.imwrite(overlay_path_abs, overlay)
                                    if success:
                                         logger.info(f"[{task_id}] Saved heatmap overlay: {overlay_filename}")
                                         return f"/{TEMP_DIR}/{overlay_filename}"
                                    else:
                                         logger.error(f"[{task_id}] Failed to save overlay image {overlay_filename} using cv2.imwrite.")
                                         return None
                                except Exception as save_e:
                                     logger.error(f"[{task_id}] Error saving overlay {overlay_filename}: {save_e}")
                                     return None
                            heatmap_url = await asyncio.to_thread(save_overlay)
                        else: logger.warning(f"[{task_id}] Overlay creation failed.")
                    else: logger.warning(f"[{task_id}] Heatmap generation failed.")

            except Exception as df_err:
                 logger.error(f"[{task_id}] Deepfake prediction/heatmap failed: {df_err}", exc_info=True)
                 deepfake_score = -1.0 # Indicate error

            analysis_data["deepfake_analysis"] = {
                "timestamp": actual_frame_time,
                "score": deepfake_score,
                "heatmap_url": heatmap_url # Include URL (or None)
            }
        else:
            logger.warning(f"[{task_id}] No suitable frame found for deepfake analysis in {os.path.basename(chunk_path)}")
            analysis_data["deepfake_analysis"] = {"timestamp": -1.0, "score": 0.0, "heatmap_url": None}

        # --- Audio Analysis ---
        # (Audio extraction and transcription logic remains largely the same)
        transcription_data = { "original": "[Audio Fail]", "detected_language": "unknown", "english": "" }
        cmd_audio = [ "ffmpeg", "-hide_banner", "-loglevel", "error", "-i", chunk_path,
                      "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", audio_path, "-y" ]
        # Consider adding -ss and -t if analyzing only a part of the audio chunk makes sense
        # e.g., -ss 1 -t 8 to skip first sec, take next 8s
        audio_proc = await asyncio.create_subprocess_exec(*cmd_audio, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        _, audio_stderr = await audio_proc.communicate()

        if audio_proc.returncode == 0:
            if await aiofiles.os.path.exists(audio_path):
                try:
                    stat_res = await aiofiles.os.stat(audio_path)
                    if stat_res.st_size > 44: # Check if file has more than just WAV header
                        logger.debug(f"[{task_id}] Transcribing audio chunk {chunk_index}")
                        # Run Whisper transcription in a separate thread
                        trans_result = await asyncio.to_thread(video_analyzer.whisper_model.transcribe, audio_path, fp16=False)
                        original_text = trans_result["text"].strip()
                        detected_lang = trans_result["language"]
                        # Translate if necessary (also in thread)
                        english_text = await translate_to_english(original_text, detected_lang) # Assuming translate_to_english handles threading
                        transcription_data = {"original": original_text, "detected_language": detected_lang, "english": english_text}
                        logger.debug(f"[{task_id}] Transcription done: Lang={detected_lang}, Text={original_text[:30]}...")
                    else:
                        logger.warning(f"[{task_id}] Audio file exists but is empty: {os.path.basename(audio_path)}")
                        transcription_data["original"] = "[No Audio Data]"
                        # Clean up empty file
                        await aiofiles.os.remove(audio_path)
                except Exception as audio_err:
                    logger.error(f"[{task_id}] Audio processing/transcription failed: {audio_err}", exc_info=True)
                    transcription_data["original"] = "[Transcription Error]"
            else:
                # FFmpeg succeeded but file is missing? Should be rare.
                logger.error(f"[{task_id}] Audio extract ffmpeg OK, but file missing: {os.path.basename(audio_path)}")
                transcription_data["original"] = "[Audio File Missing]"
        else:
            logger.warning(f"[{task_id}] Audio extract failed (Code: {audio_proc.returncode}) {os.path.basename(chunk_path)}. Stderr: {audio_stderr.decode()}")
            transcription_data["original"] = "[Audio Extraction Error]"

        analysis_data["transcription"] = transcription_data

    except Exception as e:
         logger.error(f"[{task_id}] Error during chunk analysis {chunk_path}: {e}", exc_info=True)
         # Ensure basic structure is returned on error
         analysis_data["deepfake_analysis"] = {"timestamp": -1.0, "score": -1.0, "heatmap_url": None}
         analysis_data["transcription"] = {"original": "[Analysis Failed]", "detected_language": "error", "english": ""}
         # Return None maybe? Or the error structure? Returning the structure seems better.
         # return None
    finally:
        # Release video capture if still open
        if cap and cap.isOpened():
            cap.release()
        # Clean up temporary audio file
        if await aiofiles.os.path.exists(audio_path):
            try: await aiofiles.os.remove(audio_path)
            except Exception as e_del: logger.warning(f"[{task_id}] Could not remove temp audio {audio_path}: {e_del}")
        # NOTE: Don't delete overlay here, it needs to be served. Cleanup happens later.

    logger.info(f"[{task_id}] Chunk {chunk_index} analysis finished in {time.time() - analysis_start_time:.2f}s")
    return analysis_data


async def stream_processing_manager(stream_url_platform: str):
    # (No changes needed in the manager itself)
    global stream_is_running, stream_active_tasks
    logger.info(f"MANAGER: Starting stream for {stream_url_platform}")
    stream_direct_url = None; processing_task = None
    try:
        streams = await asyncio.to_thread(streamlink_session.streams, stream_url_platform)
        stream = streams.get("best") or streams.get("worst") if streams else None
        if stream: stream_direct_url = stream.url; logger.info("MANAGER: Got direct stream URL.")
        else: raise ValueError("No suitable stream found.")
        processing_task = asyncio.create_task(stream_processing_loop(stream_direct_url), name="stream_processing_loop")
        stream_active_tasks.add(processing_task)
        await processing_task # Wait for the loop to finish or raise exception
    except asyncio.CancelledError:
         logger.info("MANAGER: Task was cancelled.")
    except Exception as e: logger.error(f"MANAGER: Error: {e}", exc_info=True)
    finally:
        logger.info("MANAGER: Stream processing finishing.")
        if processing_task in stream_active_tasks: stream_active_tasks.remove(processing_task)
        # Ensure stream_is_running is set to False if the loop exits unexpectedly
        if stream_is_running:
            logger.warning("MANAGER: Loop exited unexpectedly, ensuring stream state is stopped.")
            stream_is_running = False


async def stream_processing_loop(stream_direct_url: str):
    """
    Main asynchronous loop to download, analyze, and manage stream chunks.
    """
    global stream_is_running, stream_active_tasks # Access global state
    initial_buffer_duration = 5 # Reduced initial buffer wait
    chunk_duration = 10 # Duration of each video chunk in seconds
    playback_delay = chunk_duration * 2 # How far behind "live" playback should be
    cycle_target_time = chunk_duration # Target time for one loop iteration

    logger.info(f"LOOP: Waiting {initial_buffer_duration}s for initial stream buffer...")
    try:
        # Wait for the stream to buffer slightly at the beginning
        await asyncio.sleep(initial_buffer_duration)
    except asyncio.CancelledError:
        logger.info("LOOP: Initial sleep cancelled.")
        return # Exit if cancelled during initial wait

    # --- State Variables for the loop ---
    current_chunk_path_for_analysis = None # Path of chunk N (to be analyzed)
    next_chunk_path_downloaded = None    # Path of chunk N+1 (just downloaded)
    download_task = None                 # asyncio.Task for current download
    analysis_task = None                 # asyncio.Task for current analysis
    chunk_index = 0                      # Counter for chunks processed
    results_buffer = []                  # Buffer for completed analysis results ready for release
    stream_start_time = time.time()      # Reference time for calculating playback position
    last_successful_download_time = time.time() # Track health of download process
    max_consecutive_failures = 3         # Threshold to stop if downloads/analyses fail repeatedly
    consecutive_failures = 0             # Counter for failures

    logger.info("LOOP: Starting main processing cycle.")
    while stream_is_running:
        cycle_start_time = time.time()
        chunk_index += 1
        # Define path for the chunk to be downloaded in this cycle (N+1)
        loop_chunk_output_path = os.path.join(TEMP_DIR, f"stream_chunk_{chunk_index}.mp4")
        logger.debug(f"LOOP: Cycle {chunk_index} starting.")

        # --- 1. Start Download Task for Chunk N+1 ---
        if not stream_is_running: break # Check condition before starting task

        # Safety check: Cancel previous download if somehow still running (should be rare)
        if download_task and not download_task.done():
            logger.warning(f"LOOP: Previous download task {download_task.get_name()} still running? Cancelling.")
            download_task.cancel()
            try: await asyncio.sleep(0.1) # Allow cancellation to propagate
            except asyncio.CancelledError: pass # Ignore if loop is stopping
            if download_task in stream_active_tasks: stream_active_tasks.remove(download_task)

        # Create and start the download task
        download_task = asyncio.create_task(
            stream_download_chunk(stream_direct_url, loop_chunk_output_path, chunk_duration),
            name=f"download-{chunk_index}"
        )
        stream_active_tasks.add(download_task)
        logger.debug(f"LOOP: Started download task {download_task.get_name()}.")


        # --- 2. Schedule Analysis Task for Chunk N ---
        # If we have a path from the *previous* cycle's successful download...
        if current_chunk_path_for_analysis and stream_is_running:
            logger.debug(f"LOOP: Scheduling analysis for previous chunk {chunk_index - 1} ({os.path.basename(current_chunk_path_for_analysis)}).")

            # Safety check: Cancel previous analysis if somehow still running
            if analysis_task and not analysis_task.done():
                 logger.warning(f"LOOP: Previous analysis {analysis_task.get_name()} unfinished, cancelling.")
                 analysis_task.cancel()
                 try: await asyncio.sleep(0.1) # Allow cancellation
                 except asyncio.CancelledError: pass
                 if analysis_task in stream_active_tasks: stream_active_tasks.remove(analysis_task)

            # Create and start the analysis task
            analysis_task = asyncio.create_task(
                 stream_analyze_chunk(current_chunk_path_for_analysis, chunk_index - 1), # Analyze N-1
                 name=f"analyze-{chunk_index-1}"
             )
            stream_active_tasks.add(analysis_task)
            # Add a callback to process the result (or error) when the task completes
            # The callback function (handle_analysis_completion) needs access to results_buffer
            analysis_task.add_done_callback(
                lambda task: handle_analysis_completion(task, results_buffer)
            )
            logger.debug(f"LOOP: Scheduled analysis task {analysis_task.get_name()}.")
        elif not current_chunk_path_for_analysis and chunk_index > 1:
             logger.warning(f"LOOP: No chunk path available to schedule analysis for chunk {chunk_index - 1}.")


        # --- 3. Wait for Download Task N+1 to Complete ---
        download_success = False
        next_chunk_path_downloaded = None # Reset path for this cycle
        if download_task:
            try:
                wait_start = time.time()
                # Set a reasonable timeout for the download
                timeout_duration = cycle_target_time * 2.5 # e.g., 25 seconds for a 10s chunk
                logger.debug(f"LOOP: Waiting for download {download_task.get_name()} (timeout: {timeout_duration:.1f}s)")

                # Wait for the task, checking periodically if the stream should stop
                while time.time() - wait_start < timeout_duration:
                     if not stream_is_running:
                         if not download_task.done(): download_task.cancel() # Cancel if stopping
                         raise asyncio.CancelledError("Stream stopped during download wait")
                     if download_task.done():
                         break # Exit wait loop if task finished
                     await asyncio.sleep(0.1) # Short sleep while waiting

                # Check if the task timed out
                if not download_task.done():
                     download_task.cancel() # Cancel the overdue task
                     try: await asyncio.sleep(0.1) # Allow cancellation
                     except asyncio.CancelledError: pass
                     raise asyncio.TimeoutError(f"Download task {download_task.get_name()} timed out after {timeout_duration:.1f}s")

                # If task completed (didn't time out), get its result
                # This will re-raise any exception that occurred within the download task
                download_success = download_task.result()

                if download_success:
                    next_chunk_path_downloaded = loop_chunk_output_path # Store path for next cycle's analysis
                    last_successful_download_time = time.time() # Update health metric
                    consecutive_failures = 0 # Reset failure count on success
                    logger.debug(f"LOOP: Download {download_task.get_name()} successful.")
                else:
                    # The task finished but returned False (internal failure)
                    raise RuntimeError(f"Download task {download_task.get_name()} reported failure.")

            except (asyncio.CancelledError, asyncio.TimeoutError, RuntimeError) as e:
                # Handle expected failures gracefully
                logger.error(f"LOOP: Download {download_task.get_name()} failed or stopped: {type(e).__name__} - {e}.")
                consecutive_failures += 1
                # If the stream was explicitly stopped, don't set stream_is_running = False here
                if not isinstance(e, asyncio.CancelledError) or stream_is_running:
                     stream_is_running = False # Stop the loop on download errors/timeouts
                # Break the loop on critical download failures
                break
            except Exception as e:
                # Handle unexpected errors during the wait/result retrieval
                logger.error(f"LOOP: Unexpected error awaiting download {download_task.get_name()}: {e}", exc_info=True)
                consecutive_failures += 1
                stream_is_running = False # Stop the loop
                break
            finally:
                 # Always remove the download task from the active set once handled
                 if download_task in stream_active_tasks:
                     stream_active_tasks.remove(download_task)


        # --- 4. Release Ready Results from Buffer ---
        if stream_is_running: # Only release if we are still supposed to be running
            try:
                # Release results based on estimated playback time
                await release_buffered_results(results_buffer, stream_start_time, playback_delay, chunk_duration)
            except Exception as e:
                logger.error(f"LOOP: Error releasing results: {e}")


        # --- 5. Prep for Next Cycle & Cleanup Old Files ---
        # Set the path for the *next* iteration's analysis task (chunk N+1 becomes N)
        current_chunk_path_for_analysis = next_chunk_path_downloaded

        # --- Cleanup Logic ---
        # Calculate the index of the oldest chunk file to delete
        # Keep files for HISTORY_LIMIT items + a buffer
        chunk_to_delete_index = chunk_index - (HISTORY_LIMIT + CLEANUP_BUFFER)

        if chunk_to_delete_index > 0:
             logger.debug(f"LOOP: Checking for cleanup eligibility for index <= {chunk_to_delete_index}")
             # Define base path without extension
             base_path_to_delete = os.path.join(TEMP_DIR, f"stream_chunk_{chunk_to_delete_index}")
             path_to_delete_mp4 = f"{base_path_to_delete}.mp4"
             path_to_delete_overlay = f"{base_path_to_delete}_overlay.jpg" # Overlay filename convention

             # Schedule deletion tasks to run concurrently using asyncio.create_task
             delete_tasks = []

             # --- Asynchronously Delete MP4 (if exists) ---
             async def delete_file(f_path, f_type):
                 try:
                     if await aiofiles.os.path.exists(f_path):
                         await aiofiles.os.remove(f_path)
                         logger.info(f"LOOP: Deleted old {f_type} file: {os.path.basename(f_path)}")
                     # else: logger.debug(f"LOOP: {f_type} file {f_path} already deleted or never existed.")
                 except Exception as e_del:
                     logger.warning(f"LOOP: Failed to delete old {f_type} file {f_path}: {e_del}")

             delete_tasks.append(asyncio.create_task(delete_file(path_to_delete_mp4, "chunk")))
             delete_tasks.append(asyncio.create_task(delete_file(path_to_delete_overlay, "overlay")))

             # Optional: Wait for deletions? Usually not necessary to block the loop.
             # await asyncio.gather(*delete_tasks)
             # Deletions will run in the background.

        # --- END Cleanup Logic ---


        # --- 6. Timing & Health Check ---
        cycle_duration = time.time() - cycle_start_time
        # Calculate how long to sleep to maintain the cycle time
        sleep_time = max(0, cycle_target_time - cycle_duration)

        if cycle_duration > cycle_target_time * 1.1: # Log if cycle takes >10% longer than target
            logger.warning(f"LOOP: Cycle {chunk_index} duration ({cycle_duration:.2f}s) exceeded target ({cycle_target_time:.1f}s).")
            sleep_time = 0 # Don't sleep if the loop is already falling behind

        # --- Health Checks ---
        # Check if downloads have been failing for too long
        # Allow ~4 missed chunks before giving up
        if time.time() - last_successful_download_time > cycle_target_time * (max_consecutive_failures + 1):
             logger.error(f"LOOP: No successful download in ~{cycle_target_time * (max_consecutive_failures + 1)}s. Stopping stream.")
             stream_is_running = False # Signal stop
             break # Exit loop

        # Check if consecutive failures threshold reached
        if consecutive_failures >= max_consecutive_failures:
             logger.error(f"LOOP: Stopping stream after {consecutive_failures} consecutive failures.")
             stream_is_running = False # Signal stop
             break # Exit loop

        # --- Sleep ---
        if sleep_time > 0 and stream_is_running:
            try:
                # logger.debug(f"LOOP: Sleeping for {sleep_time:.2f}s")
                await asyncio.sleep(sleep_time)
            except asyncio.CancelledError:
                logger.info("LOOP: Cycle sleep cancelled.")
                break # Exit loop cleanly if cancelled during sleep

    # --- Loop Exit ---
    logger.info("LOOP: Exiting main processing loop.")

    # --- Final Cleanup Attempt ---
    # Cancel any remaining analysis task that might be running when the loop exits
    if analysis_task and not analysis_task.done():
        logger.info(f"LOOP: Cancelling final analysis task {analysis_task.get_name()} on exit.")
        analysis_task.cancel()
        try:
             # Give it a very short time to finish cancellation
             await asyncio.wait_for(analysis_task, timeout=1.0)
        except (asyncio.TimeoutError, asyncio.CancelledError):
             logger.debug(f"LOOP: Timeout or cancellation during final cleanup of task {analysis_task.get_name()}.")
             pass # Ignore timeout/cancel on final cleanup
        except Exception as e:
             # Log any other error during final cancellation
             logger.warning(f"LOOP: Error during final analysis task cancellation ({analysis_task.get_name()}): {e}")
        finally:
            # Ensure it's removed from the active set
            if analysis_task in stream_active_tasks:
                stream_active_tasks.remove(analysis_task)

    # Download task should have been handled within the loop's wait block or finally clause
    logger.info("LOOP: Finished cleanup on exit.")

def handle_analysis_completion(task: asyncio.Task, results_buffer: List):
    # (No changes needed here, it just passes the result dict which now might contain heatmap_url)
    global stream_fact_check_buffer, stream_last_fact_check_time, fact_checker, stream_active_tasks
    analysis_results_dict = None
    chunk_index = -1
    task_name = task.get_name()

    try:
        # Extract chunk index from task name (e.g., "analyze-123")
        try: chunk_index = int(task_name.split('-')[-1])
        except (IndexError, ValueError): logger.error(f"ANALYSIS_CB: Could not parse chunk index from task name '{task_name}'"); return

        if task.cancelled():
            logger.warning(f"ANALYSIS_CB: Task {task_name} was cancelled."); return

        # Check for exceptions during analysis
        exc = task.exception()
        if exc:
            logger.error(f"ANALYSIS_CB: Task {task_name} failed with exception: {exc}", exc_info=exc)
            # Optionally add an error marker to results buffer?
            # results_buffer.append({"chunk_index": chunk_index, "error": str(exc)}) # Example error marker
            return

        # Get the result dictionary from the task
        analysis_results_dict = task.result()

        if analysis_results_dict:
            logger.info(f"ANALYSIS_CB: Processing analysis result for chunk {chunk_index}")
            # Extract transcription for fact-checking buffer
            transcription = analysis_results_dict.get("transcription", {}).get("english", "").strip()
            if transcription and transcription != "[Analysis Pending]" and not transcription.startswith("["): # Avoid adding error messages
                stream_fact_check_buffer += " " + transcription

            # Construct the final result object to be sent/buffered
            final_result = {
                "video_chunk_url": f"/{TEMP_DIR}/stream_chunk_{chunk_index}.mp4",
                "chunk_index": chunk_index,
                "analysis_timestamp": time.time(), # Timestamp when analysis finished processing
                "deepfake_analysis": analysis_results_dict.get("deepfake_analysis"), # Includes score & heatmap_url
                "transcription": analysis_results_dict.get("transcription"),
                "fact_check_results": [],
                "fact_check_context_current": False
            }

            # Periodic Fact-Checking (Synchronous call within the callback)
            current_time = time.time()
            run_fact_check = (
                stream_fact_check_buffer.strip() and
                (current_time - stream_last_fact_check_time >= STREAM_FACT_CHECK_BUFFER_DURATION)
            )

            if run_fact_check:
                logger.info(f"ANALYSIS_CB: Running fact check on buffered text (length {len(stream_fact_check_buffer)})...")
                # Copy buffer and reset global state immediately
                buffer_to_check = stream_fact_check_buffer.strip()
                stream_fact_check_buffer = ""
                stream_last_fact_check_time = current_time

                try:
                     # Run fact-checking (this is blocking)
                     # Consider if this should be run in a separate thread if it becomes a bottleneck
                     fc_result = fact_checker.check(buffer_to_check, num_workers=2) # Use fact_checker instance
                     processed_claims = fc_result.get("processed_claims", [])
                     final_result["fact_check_results"] = processed_claims
                     final_result["fact_check_context_current"] = True # Mark this result contains current FC data
                     logger.info(f"ANALYSIS_CB: Fact check completed. Found {len(processed_claims)} claims.")
                except Exception as fc_e:
                     logger.error(f"ANALYSIS_CB: Fact check execution failed: {fc_e}", exc_info=True)
                     final_result["fact_check_results"] = [{"error": f"Fact Check Failed: {fc_e}"}]
                     final_result["fact_check_context_current"] = True # Still mark as current context attempt

            # Insert the result into the buffer, maintaining sorted order by chunk_index
            # This ensures results are released chronologically even if analysis finishes out of order
            ins_idx = 0
            while ins_idx < len(results_buffer) and results_buffer[ins_idx]["chunk_index"] < chunk_index:
                ins_idx += 1
            results_buffer.insert(ins_idx, final_result)

            logger.debug(f"ANALYSIS_CB: Added chunk {chunk_index} result. Buffer size: {len(results_buffer)}")
        else:
            # Handle case where analysis task returned None (should be rare if errors are handled)
            logger.warning(f"ANALYSIS_CB: Analysis task {task_name} returned None result.")

    except Exception as cb_err:
        logger.error(f"ANALYSIS_CB: Error in callback for task {task_name}: {cb_err}", exc_info=True)
    finally:
        # Ensure task is removed from the active set regardless of outcome
        if task in stream_active_tasks:
            stream_active_tasks.remove(task)


async def release_buffered_results(results_buffer: List, stream_start_time: float, playback_delay: int, chunk_duration: int):
    # (No changes needed here)
    global stream_result_queue
    if not results_buffer: return

    # Calculate the index of the chunk that *should* be playing now
    elapsed_time = time.time() - stream_start_time
    # The chunk index that is ready to be released (finished playback_delay ago)
    ready_chunk_index_threshold = int(max(0, elapsed_time - playback_delay) / chunk_duration)

    # Release all chunks in the buffer that are at or before the threshold
    while results_buffer and results_buffer[0]["chunk_index"] <= ready_chunk_index_threshold:
        result_to_release = results_buffer.pop(0) # Get the oldest ready chunk
        try:
            # Put result onto the queue for websockets, with a timeout
            await asyncio.wait_for(stream_result_queue.put(result_to_release), timeout=2.0)
            logger.info(f"RELEASE: Sent chunk {result_to_release['chunk_index']} results to WS queue (QSize: {stream_result_queue.qsize()})")
        except asyncio.TimeoutError:
            # If queue is full (e.g., WS client disconnected/slow), log error and put it back
            logger.error(f"RELEASE: Timeout putting chunk {result_to_release['chunk_index']} onto WS queue. Re-inserting into buffer.")
            results_buffer.insert(0, result_to_release) # Put it back at the front
            break # Stop trying to release more chunks for now if queue is blocked
        except Exception as e:
            # Handle other potential queue errors
            logger.error(f"RELEASE: Failed to put chunk {result_to_release['chunk_index']} onto WS queue: {e}")
            results_buffer.insert(0, result_to_release) # Put it back
            break # Stop releasing on other errors too

# === FastAPI Endpoints ===

# --- STREAM Endpoints ---
# (No changes needed for start, results WS, stop endpoints themselves)
@app.post("/api/stream/analyze")
async def start_stream_analysis(data: dict):
    global stream_is_running, stream_active_tasks, stream_fact_check_buffer, stream_last_fact_check_time, stream_result_queue
    url = data.get("url")
    if not url: raise HTTPException(status_code=400, detail="Stream URL required")
    if stream_is_running:
        logger.warning("Start request received but stream is already running.")
        return {"message": "Stream analysis is already running."}

    logger.info(f"Start analysis request received for URL: {url}")

    # --- Reset State ---
    stream_is_running = True
    stream_fact_check_buffer = ""
    stream_last_fact_check_time = time.time() # Reset fact check timer

    # Clear the result queue
    while not stream_result_queue.empty():
        try: stream_result_queue.get_nowait(); stream_result_queue.task_done()
        except asyncio.QueueEmpty: break
        except Exception as e: logger.warning(f"Error clearing result queue item: {e}")

    # Cancel any lingering tasks from a previous run (shouldn't happen if stop works)
    if stream_active_tasks:
         logger.warning(f"Found {len(stream_active_tasks)} active tasks before start. Cancelling them.")
         tasks_to_cancel = list(stream_active_tasks)
         for task in tasks_to_cancel: task.cancel()
         try:
             # Give a short time for tasks to acknowledge cancellation
             await asyncio.wait_for(asyncio.gather(*tasks_to_cancel, return_exceptions=True), timeout=1.0)
         except asyncio.TimeoutError: logger.warning("Timeout waiting for old tasks to cancel during start.")
         stream_active_tasks.clear() # Ensure set is empty

    # --- Start Manager ---
    # Create the main processing manager task
    asyncio.create_task(stream_processing_manager(url), name="stream_processing_manager")
    logger.info("Stream processing manager task created.")
    return {"message": "Stream analysis starting..."}

@app.websocket("/api/stream/results")
async def stream_results_websocket(websocket: WebSocket):
    await websocket.accept()
    client_id = f"{websocket.client.host}:{websocket.client.port}"
    logger.info(f"WebSocket client connected: {client_id}")
    queue_get_task = None
    try:
        while True:
            # Use asyncio.create_task to allow breaking loop if websocket closes while waiting
            queue_get_task = asyncio.create_task(stream_result_queue.get())
            done, pending = await asyncio.wait(
                {queue_get_task, asyncio.create_task(websocket.receive_text())}, # Also wait for potential close message
                return_when=asyncio.FIRST_COMPLETED
            )

            if queue_get_task in done:
                 result = queue_get_task.result()
                 try:
                     await websocket.send_json(result)
                     stream_result_queue.task_done() # Mark task done only after successful send
                 except Exception as send_err:
                      logger.error(f"WS Send Error to {client_id}: {send_err}. Client likely disconnected.")
                      # Don't mark task done, it wasn't processed by this client.
                      # Re-queue? No, might lead to infinite loop if client never recovers.
                      # Best to just break and let the next client get it (if any).
                      stream_result_queue.task_done() # Or maybe mark done to avoid blocking? Let's mark done.
                      break # Exit loop on send error
            else:
                 # The websocket.receive_text() task completed, likely a close message or error
                 logger.info(f"WS client {client_id} sent message or disconnected while waiting for queue.")
                 queue_get_task.cancel() # Cancel the pending queue.get()
                 await asyncio.sleep(0) # Allow cancellation to register
                 break # Exit loop

    except asyncio.CancelledError:
        logger.info(f"WebSocket task for {client_id} cancelled.")
        if queue_get_task and not queue_get_task.done(): queue_get_task.cancel()
    except Exception as e:
        # Catch generic exceptions (like WebSocketDisconnect)
        logger.info(f"WebSocket client {client_id} disconnected: {type(e).__name__} - {e}")
        if queue_get_task and not queue_get_task.done(): queue_get_task.cancel()
    finally:
        logger.info(f"WebSocket connection closing for {client_id}")
        # Ensure task done is called if we broke mid-process? Risky.
        # WebSocket close handled automatically by FastAPI/Starlette context manager
        # await websocket.close() # Usually not needed here


@app.post("/api/stream/stop")
async def stop_stream_analysis():
    global stream_is_running, stream_active_tasks, stream_result_queue
    if not stream_is_running:
        logger.info("Stop request received but stream is not running.")
        return {"message": "Stream analysis is not running."}

    logger.info("Stop analysis request received.")
    stream_is_running = False # Signal loops and tasks to stop

    logger.info(f"Cancelling {len(stream_active_tasks)} active stream tasks...")
    # Create a list of tasks to cancel *before* iterating
    tasks_to_cancel = list(stream_active_tasks)
    for task in tasks_to_cancel:
        task.cancel()

    # Wait for tasks to finish cancellation (with timeout)
    if tasks_to_cancel:
        try:
            # Gather waits for all tasks, return_exceptions stops it from failing on first cancelled task
            await asyncio.wait_for(asyncio.gather(*tasks_to_cancel, return_exceptions=True), timeout=5.0)
            logger.info("Active stream tasks cancelled.")
        except asyncio.TimeoutError:
            logger.warning("Timeout waiting for stream tasks to complete cancellation.")
        except Exception as e:
             logger.error(f"Error during task cancellation gathering: {e}")

    stream_active_tasks.clear() # Clear the set after attempting cancellation

    # Clear the result queue after stopping tasks
    logger.info("Clearing stream result queue...")
    cleared_count = 0
    while not stream_result_queue.empty():
        try:
            stream_result_queue.get_nowait()
            stream_result_queue.task_done()
            cleared_count += 1
        except asyncio.QueueEmpty:
            break
        except Exception as e:
            logger.warning(f"Error clearing stream result queue item: {e}")
            # Break here to avoid potential infinite loop if task_done fails repeatedly
            break
    logger.info(f"Cleared {cleared_count} items from result queue.")

    logger.info("Stream analysis stopped successfully.")
    return {"message": "Stream analysis stopped"}


# === UPLOAD MODE ENDPOINTS and HELPERS ===

async def broadcast_progress(progress: float):
    # (No changes needed here)
    disconnected_clients = []
    message = json.dumps({"progress": progress})
    # Iterate over a copy of the list to allow safe removal
    for client in list(upload_progress_clients):
        try:
            await client.send_text(message)
        except Exception as e:
            # Log specific exception type if useful (e.g., WebSocketDisconnect)
            logger.warning(f"Progress send failed ({type(e).__name__}). Removing client.")
            disconnected_clients.append(client)

    # Remove disconnected clients from the main list
    for client in disconnected_clients:
        if client in upload_progress_clients:
            try:
                 upload_progress_clients.remove(client)
            except ValueError:
                 pass # Ignore if already removed somehow


# --- Translate functions (no changes needed) ---
async def translate_to_language(text: str, target_language: str) -> str:
    if not text or not GROQ_API_KEY: return text
    # Check if source and target are the same (case-insensitive)
    if target_language.lower() == "english" and text.strip() == text: # Simple check for already English
        # A more robust check might involve language detection if needed
        return text

    logger.debug(f"Translating to {target_language} using Groq...")
    try:
        # Use asyncio.to_thread for the blocking requests call
        def do_translate():
            url = "https://api.groq.com/openai/v1/chat/completions"
            headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
            payload = {
                "model": "llama3-8b-8192", # Consider making model configurable
                "messages": [
                    {"role": "system", "content": f"Translate the following text accurately to {target_language}. Output *only* the translated text, without any introductory phrases, explanations, or quotation marks."},
                    {"role": "user", "content": text}
                ],
                "max_tokens": 2048, # Adjust based on expected length
                "temperature": 0.1 # Low temp for more deterministic translation
            }
            try:
                response = requests.post(url, headers=headers, json=payload, timeout=30) # 30s timeout
                response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
                result = response.json()
                translated_content = result.get("choices", [{}])[0].get("message", {}).get("content", "")
                # Clean up potential markdown or extra quotes (though prompt aims to avoid this)
                return translated_content.strip().strip('"').strip("'").strip()
            except requests.exceptions.Timeout:
                logger.error(f"Translation request timed out after 30 seconds.")
                return f"[Translation Timeout to {target_language}]"
            except requests.exceptions.RequestException as req_err:
                logger.error(f"Translation API request failed: {req_err}")
                return f"[Translation API Error to {target_language}]"
            except (KeyError, IndexError, AttributeError) as json_err:
                 logger.error(f"Failed to parse translation API response: {json_err}. Response: {result if 'result' in locals() else 'N/A'}")
                 return f"[Translation Response Parse Error to {target_language}]"

        translated = await asyncio.to_thread(do_translate)
        logger.debug(f"Translation result (truncated): {translated[:50]}...")
        return translated
    except Exception as e:
        # Catch errors in the asyncio/threading logic itself
        logger.error(f"Unexpected error during translation to {target_language}: {e}", exc_info=True)
        return f"[Translation System Error to {target_language}]" # Return original text on failure

async def translate_to_english(transcription: str, source_language: str) -> str:
    # Normalize language codes/names if possible (e.g., 'en-US' -> 'en')
    # Basic check:
    if source_language.lower().startswith("en") or not transcription:
        return transcription
    logger.info(f"Translating from '{source_language}' to English...")
    return await translate_to_language(transcription, "English")


@app.post("/api/video/translate")
async def translate_transcription_endpoint(data: dict):
    # (No changes needed here)
    transcription = data.get("transcription")
    target_language = data.get("language", "en") # Default to English
    if not transcription:
        raise HTTPException(status_code=400, detail="Transcription text required")
    if not target_language:
         raise HTTPException(status_code=400, detail="Target language required")

    try:
        translated_text = await translate_to_language(transcription, target_language)
        return {"translation": translated_text}
    except Exception as e:
        logger.error(f"Translation endpoint error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Translation failed: {str(e)}")

@app.websocket("/api/video/progress")
async def video_progress_websocket(websocket: WebSocket):
    # (No changes needed here)
    await websocket.accept()
    client_id = f"{websocket.client.host}:{websocket.client.port}"
    logger.info(f"Upload progress client connected: {client_id}")
    upload_progress_clients.append(websocket)
    try:
        # Keep connection open, wait for disconnect
        while True:
            # Keepalive or wait for close
            await websocket.receive_text() # Wait for message (or disconnect)
    except Exception as e:
        # Handles WebSocketDisconnect, CancelledError etc.
        logger.info(f"Upload progress client {client_id} disconnected: {type(e).__name__}")
    finally:
        logger.info(f"Upload progress client connection closed: {client_id}")
        if websocket in upload_progress_clients:
            try: upload_progress_clients.remove(websocket)
            except ValueError: pass # Ignore if already removed

@app.post("/api/video/analyze")
async def analyze_video_upload(file: UploadFile = File(...)):
    # --- Updated to handle new return structure from VideoAnalyzer ---
    # Use a timestamp for uniqueness, sanitize filename
    timestamp = int(time.time())
    safe_filename = "".join(c for c in file.filename if c.isalnum() or c in ('_', '-')).rstrip()
    temp_path = os.path.join(TEMP_DIR, f"upload_{timestamp}_{safe_filename}")
    logger.info(f"Upload request: {file.filename} -> {temp_path}")

    # Ensure progress starts at 0
    await broadcast_progress(0.0)

    try:
        # Async save using aiofiles
        async with aiofiles.open(temp_path, "wb") as f:
            # Read file in chunks to handle large uploads without excessive memory use
            while content := await file.read(1024 * 1024): # Read 1MB chunks
                 await f.write(content)
        # Verify file saved
        if not await aiofiles.os.path.exists(temp_path):
             raise IOError(f"Failed to save uploaded file to {temp_path}")
        stat_res = await aiofiles.os.stat(temp_path)
        logger.info(f"Upload saved: {temp_path}, size: {stat_res.st_size} bytes")
        await broadcast_progress(0.05) # Progress after save

        # --- Direct Analysis Path ---
        logger.info(f"Starting direct analysis (incl. heatmaps): {os.path.basename(temp_path)}")
        await broadcast_progress(0.1) # Progress before analysis starts

        # Define progress callback function for VideoAnalyzer
        async def upload_progress_update(analyzer_progress: float):
             # Scale analyzer progress (0-1) to the backend progress range (0.1 to 0.95)
             backend_progress = 0.1 + analyzer_progress * 0.85
             await broadcast_progress(backend_progress)

        # Call VideoAnalyzer.analyze_video in a thread
        # It now returns: transcription, final_score, frames_data (with overlays), detected_language
        analysis_tuple = await asyncio.to_thread(
            video_analyzer.analyze_video,
            temp_path,
            upload_progress_update # Pass the async callback
        )
        transcription, final_score, frames_data, detected_language = analysis_tuple

        logger.info(f"Direct video analysis done. Score: {final_score:.3f}, Lang: {detected_language}")
        # Note: Progress is handled by the callback now

        # Translate to English if needed (after video analysis)
        english_transcription = await translate_to_english(transcription, detected_language)
        await broadcast_progress(0.96) # Progress after translation

        # Run fact-check and text analysis (these don't have fine-grained progress)
        logger.info("Running fact-check...")
        fact_check_result = await asyncio.to_thread(fact_checker.check, english_transcription, num_workers=2)
        await broadcast_progress(0.97)

        logger.info("Running text analysis...")
        text_analysis_result = await text_analyzer.analyze_text(english_transcription) # Assuming this is async or wrapped
        await broadcast_progress(0.99)

        # --- Prepare Final Response ---
        response = {
            "original_transcription": transcription,
            "detected_language": detected_language,
            "english_transcription": english_transcription,
            "final_score": final_score, # Overall max deepfake score
            "frames_data": frames_data, # Contains timestamps, scores, indices, overlays
            "text_analysis": {
                "political_bias": text_analysis_result.political_bias,
                "emotional_triggers": text_analysis_result.emotional_triggers,
                "stereotypes": text_analysis_result.stereotypes,
                "manipulation_score": text_analysis_result.manipulation_score,
                "entities": text_analysis_result.entities,
                "locations": text_analysis_result.locations, # Passed through if available
                "fact_check_result": fact_check_result
            }
        }
        await broadcast_progress(1.0) # Final progress update
        logger.info(f"Analysis complete for {file.filename}.")
        return JSONResponse(content=response)

    except FileNotFoundError as fnf_err:
        logger.error(f"File not found during upload analysis: {fnf_err}")
        await broadcast_progress(-1.0) # Signal error
        raise HTTPException(status_code=404, detail=str(fnf_err))
    except IOError as io_err:
         logger.error(f"File save/read error during upload analysis: {io_err}")
         await broadcast_progress(-1.0)
         raise HTTPException(status_code=500, detail=f"File handling error: {io_err}")
    except Exception as e:
        logger.error(f"Error analyzing upload {file.filename}: {e}", exc_info=True)
        await broadcast_progress(-1.0) # Signal error
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")
    finally:
        # Ensure temporary upload file is deleted
        if 'temp_path' in locals() and await aiofiles.os.path.exists(temp_path):
            try:
                await aiofiles.os.remove(temp_path)
                logger.info(f"Deleted temp upload file: {temp_path}")
            except Exception as e_del:
                logger.error(f"Failed to delete temp upload file {temp_path}: {e_del}")


# === SHUTDOWN ===
@app.on_event("shutdown")
async def shutdown_event():
    global stream_is_running, stream_active_tasks
    logger.info("Application shutdown initiated.")

    # --- Stop Stream Gracefully ---
    stream_is_running = False # Signal stream loops to stop
    if stream_active_tasks:
        logger.info(f"Shutdown: Cancelling {len(stream_active_tasks)} active stream tasks...")
        tasks_to_cancel = list(stream_active_tasks)
        for task in tasks_to_cancel:
            task.cancel()
        try:
            await asyncio.wait_for(asyncio.gather(*tasks_to_cancel, return_exceptions=True), timeout=3.0)
            logger.info("Stream tasks cancellation complete.")
        except asyncio.TimeoutError:
            logger.warning("Timeout waiting for stream tasks to cancel during shutdown.")
        except Exception as e:
             logger.error(f"Error during stream task cancellation gathering: {e}")
        stream_active_tasks.clear()

    # --- Close External Connections ---
    # Close Kafka Producer (if used)
    if video_producer:
        try:
            logger.info("Closing Kafka producer...")
            await asyncio.to_thread(video_producer.close) # Ensure it's run in thread if blocking
            logger.info("Kafka producer closed.")
        except Exception as e:
            logger.error(f"Error closing Kafka Producer: {e}")

    # Close Neo4j (FactChecker handles this)
    try:
        logger.info("Closing Neo4j connection via FactChecker...")
        # Assuming fact_checker.close_neo4j() is thread-safe or non-blocking
        # If it's blocking, wrap in asyncio.to_thread
        await asyncio.to_thread(fact_checker.close_neo4j)
        logger.info("Neo4j connection closed.")
    except Exception as e:
        logger.error(f"Error closing Neo4j connection: {e}", exc_info=True)

    # --- Async Cleanup of Temp Directory ---
    logger.info(f"Shutdown: Cleaning up temp directory: {TEMP_DIR}")
    cleanup_tasks = []
    try:
        if await aiofiles.os.path.isdir(TEMP_DIR):
            async for filename in aiofiles.os.listdir(TEMP_DIR):
                file_path = os.path.join(TEMP_DIR, filename)
                # Check if it's a file before adding delete task
                try:
                     if await aiofiles.os.path.isfile(file_path):
                          # Schedule deletion task for files (mp4, wav, jpg overlays etc.)
                          cleanup_tasks.append(
                              asyncio.create_task(aiofiles.os.remove(file_path), name=f"delete-{filename}")
                          )
                     else:
                          logger.debug(f"Skipping cleanup for non-file item: {filename}")
                except Exception as stat_err:
                     logger.warning(f"Error stating file {file_path} during cleanup: {stat_err}")

            if cleanup_tasks:
                 logger.info(f"Attempting to delete {len(cleanup_tasks)} files from {TEMP_DIR}...")
                 # Wait for all deletion tasks to complete
                 results = await asyncio.gather(*cleanup_tasks, return_exceptions=True)
                 # Log any errors during deletion
                 success_count = 0
                 for i, result in enumerate(results):
                     task_name = cleanup_tasks[i].get_name()
                     if isinstance(result, Exception):
                         logger.error(f"Failed to delete file during cleanup ({task_name}): {result}")
                     else:
                          success_count += 1
                 logger.info(f"Temp directory cleanup finished. Deleted {success_count}/{len(cleanup_tasks)} files.")
            else:
                 logger.info("Temp directory empty or contained no files, no cleanup needed.")
    except FileNotFoundError:
        logger.info(f"Temp directory {TEMP_DIR} not found, skipping cleanup.")
    except Exception as e_clean:
        logger.error(f"Error during temp directory cleanup: {e_clean}", exc_info=True)

    logger.info("Application shutdown sequence finished.")


# === MAIN GUARD ===
if __name__ == "__main__":
    import uvicorn
    logger.info("Starting Uvicorn server on http://127.0.0.1:5001")
    # Use log_config=None to rely on Python's logging setup
    uvicorn.run(
        "main:app",
        host="127.0.0.1",
        port=5001,
        log_config=None, # Use our configured logger
        # log_level="info", # Can be set, but log_config=None is preferred
        reload=True # Keep reload for development
    )