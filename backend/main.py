# backend/main.py
from fastapi import FastAPI, UploadFile, File, WebSocket, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
# --- Keep imports needed for BOTH modes ---
from video_analyzer import VideoAnalyzer # Used by both
from analyzer import OptimizedAnalyzer # Used by both (or parts of it)
from optimized_deepfake_detector import OptimizedDeepfakeDetector # Used by stream, potentially by upload? Check analyze_video
from knowledge_graph import KnowledgeGraphManager # Used by both (within Analyzer/FactChecker)
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
from typing import Dict, Any, List

# --- Imports specifically for UPLOAD mode ---
from video_processor import VideoProducer # For Kafka in upload
from spark_video_processor import SparkVideoProcessor # If Spark is used for upload analysis
from kafka import KafkaConsumer # For consuming Spark results in upload

# --- Imports specifically for STREAM mode ---
from streamlink import Streamlink

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
# Ensure FactChecker or Analyzer load their own keys internally if needed
# FACT_CHECK_API_KEY = os.getenv("FACT_CHECK_API_KEY") # Commented out if managed internally by FC/Analyzer

logger = logging.getLogger(__name__)
# Consistent logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(threadName)s] - %(message)s',
    handlers=[logging.StreamHandler(), logging.FileHandler('veristream_backend.log')] # Log to console and file
)

# Suppress Hugging Face parallelism warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Ensure temp directory exists
TEMP_DIR = "temp_media" # Use a different name to avoid conflicts? Or manage subdirs
if not os.path.exists(TEMP_DIR):
    os.makedirs(TEMP_DIR)

app = FastAPI()
# Mount the temp directory for serving files (both chunks and potentially upload temps)
app.mount(f"/{TEMP_DIR}", StaticFiles(directory=TEMP_DIR), name="temp_files")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"], # Adjust as needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Instantiate components used by BOTH modes ---
# Note: Loading models takes time, done once on startup.
logger.info("Initializing models and components...")
try:
    video_analyzer = VideoAnalyzer() # Has Whisper
    text_analyzer = OptimizedAnalyzer(use_gpu=True) # Has text models, KG integration
    fact_checker = FactChecker() # Standalone fact-checking logic + Neo4j
    deepfake_detector = OptimizedDeepfakeDetector() # Standalone deepfake model
    # --- Components specifically for UPLOAD mode ---
    video_producer = VideoProducer() # Kafka producer for uploads
    # spark_processor = SparkVideoProcessor() # Uncomment if Spark processing for uploads is active
    # --- Components specifically for STREAM mode ---
    streamlink_session = Streamlink()
    logger.info("Models and components initialized.")
except Exception as e:
     logger.critical(f"FATAL: Failed to initialize core components: {e}", exc_info=True)
     # Optionally exit or run in a degraded state?
     raise RuntimeError(f"Core component initialization failed: {e}") from e


# --- Global State for STREAM mode ---
stream_is_running = False
stream_result_queue = asyncio.Queue()
stream_fact_check_buffer = ""
STREAM_FACT_CHECK_BUFFER_DURATION = 30 # Check every 30 seconds of text
stream_last_fact_check_time = 0
stream_active_tasks = set() # Keep track of running stream tasks for cleanup

# --- Global State for UPLOAD mode ---
upload_progress_clients = []


# === STREAM MODE FUNCTIONS ===

async def stream_download_chunk(stream_url: str, output_path: str, duration: int) -> bool:
    """Downloads a stream chunk using FFmpeg."""
    task_id = asyncio.current_task().get_name() if asyncio.current_task() else 'download'
    logger.info(f"[{task_id}] Starting download: {os.path.basename(output_path)} for {duration}s")
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "warning", # Show warnings for issues
        "-i", stream_url,
        "-t", str(duration),
        # Video encoding: H.264, very fast preset, reasonable quality (adjust crf higher for lower quality/faster)
        "-c:v", "libx264", "-preset", "ultrafast", "-crf", "28",
         # Ensure pixel format is browser-compatible (often needed)
        "-pix_fmt", "yuv420p",
        # Audio encoding: AAC, standard bitrate
        "-c:a", "aac", "-b:a", "128k",
        # Force output format
        "-f", "mp4",
        output_path,
        "-y" # Overwrite existing file
    ]
    process = await asyncio.create_subprocess_exec(
        *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
    )
    stdout, stderr = await process.communicate()

    if process.returncode != 0:
        stderr_str = stderr.decode()
        logger.error(f"[{task_id}] FFmpeg re-encode download failed for {os.path.basename(output_path)} (Code: {process.returncode}): {stderr_str}")
        return False
    else:
        # Check if file exists and has size after encoding
        if os.path.exists(output_path) and os.path.getsize(output_path) > 100: # Check for minimal size
            logger.info(f"[{task_id}] Finished download (re-encoded): {os.path.basename(output_path)}")
            return True
        else:
            logger.error(f"[{task_id}] FFmpeg process finished for {os.path.basename(output_path)} but output file is missing or empty.")
            return False # Failed for other reasons


async def stream_analyze_chunk(chunk_path: str, chunk_index: int) -> Dict[str, Any] | None:
    """Analyzes a single 10-second video chunk (stream mode)."""
    task_id = asyncio.current_task().get_name() if asyncio.current_task() else f'analyze-{chunk_index}'
    if not os.path.exists(chunk_path):
        logger.warning(f"[{task_id}] Chunk file not found for analysis: {chunk_path}")
        return None

    logger.info(f"[{task_id}] Analyzing chunk: {os.path.basename(chunk_path)}")
    analysis_start_time = time.time()
    analysis_data = {}
    analysis_frame = None
    actual_frame_time = -1
    cap = None
    audio_path = os.path.join(TEMP_DIR, f"stream_audio_{chunk_index}.wav") # Ensure audio temp in TEMP_DIR

    try:
        # --- Video Analysis (Deepfake Sampling) ---
        cap = cv2.VideoCapture(chunk_path)
        if not cap.isOpened():
            logger.error(f"[{task_id}] Failed to open chunk for analysis: {chunk_path}")
            return None

        target_frame_time_ms = 5 * 1000 # Analyze frame around 5s mark

        while True: # Use break instead of cap.isOpened() for clarity
            ret, frame = cap.read()
            if not ret: break
            current_time_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
            # Method #2: Analysis Offset - Skip first second
            if current_time_ms > 1000:
                if analysis_frame is None and current_time_ms >= target_frame_time_ms:
                    analysis_frame = frame.copy() # Important: copy frame data
                    actual_frame_time = current_time_ms / 1000.0
                    logger.debug(f"[{task_id}] Selected frame for deepfake at ~{actual_frame_time:.2f}s")
                    break # Found frame, exit loop

        deepfake_score = 0.0
        if analysis_frame is not None:
            resized_frame = cv2.resize(analysis_frame, (640, 480))
            # Run prediction in executor to avoid blocking event loop if it's CPU intensive
            # Assuming predict_batch is thread-safe or releases GIL appropriately
            # If predict_batch is pure CPU and blocks, use to_thread
            # If it involves IO or GPU waits, it might be okay directly
            # Let's assume it's okay for now, or wrap if performance issues arise
            scores = deepfake_detector.predict_batch([resized_frame])
            deepfake_score = scores[0] if scores else 0.0
            analysis_data["deepfake_analysis"] = {"timestamp": actual_frame_time, "score": deepfake_score}
        else:
            logger.warning(f"[{task_id}] Could not find frame near 5s mark in {os.path.basename(chunk_path)}")
            analysis_data["deepfake_analysis"] = {"timestamp": -1, "score": 0.0}

        # --- Audio Analysis (Transcription) ---
        transcription_data = { "original": "[Audio Fail]", "detected_language": "unknown", "english": "" }
        cmd_audio = [
            "ffmpeg", "-hide_banner", "-loglevel", "error",
            "-i", chunk_path, "-ss", "1", "-t", "9", # Offset + duration
            "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
            audio_path, "-y"
        ]
        audio_proc = await asyncio.create_subprocess_exec(
            *cmd_audio, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )
        _, audio_stderr = await audio_proc.communicate()

        if audio_proc.returncode != 0:
             logger.error(f"[{task_id}] FFmpeg audio extract failed for {os.path.basename(chunk_path)}: {audio_stderr.decode()}")
        elif os.path.exists(audio_path) and os.path.getsize(audio_path) > 44:
            logger.debug(f"[{task_id}] Transcribing audio for chunk {chunk_index}")
            # Run Whisper in executor thread
            trans_result = await asyncio.to_thread(video_analyzer.whisper_model.transcribe, audio_path)
            original_text = trans_result["text"].strip()
            detected_lang = trans_result["language"]
            english_text = await translate_to_english(original_text, detected_lang)
            transcription_data = {"original": original_text, "detected_language": detected_lang, "english": english_text}
            logger.debug(f"[{task_id}] Transcription: Lang={detected_lang}, Text={original_text[:30]}...")
        else:
            logger.warning(f"[{task_id}] Audio file empty/not created for {os.path.basename(chunk_path)}")
            transcription_data["original"] = "[No Audio]"

        analysis_data["transcription"] = transcription_data

    except Exception as e:
         logger.error(f"[{task_id}] Error during chunk analysis for {chunk_path}: {e}", exc_info=True)
         # Return partial data or None? Let's return None on major failure.
         return None # Indicate failure
    finally:
        if cap and cap.isOpened(): cap.release()
        if os.path.exists(audio_path):
            try: os.remove(audio_path)
            except Exception as e: logger.warning(f"[{task_id}] Could not remove temp audio {audio_path}: {e}")

    logger.info(f"[{task_id}] Chunk {chunk_index} analysis took {time.time() - analysis_start_time:.2f}s")
    return analysis_data

# Wrapper to run the stream processing loop and manage tasks
async def stream_processing_manager(stream_url_platform: str):
    global stream_is_running, stream_fact_check_buffer, stream_last_fact_check_time, stream_active_tasks

    logger.info(f"MANAGER: Starting stream processing for {stream_url_platform}")
    stream_direct_url = None
    processing_task = None # The main loop task

    try:
        # Get stream URL first
        streams = streamlink_session.streams(stream_url_platform)
        if streams:
            stream = streams.get("best") or streams.get("worst")
            if stream:
                stream_direct_url = stream.url
                logger.info(f"MANAGER: Obtained direct stream URL: {stream_direct_url[:50]}...")
            else: raise ValueError("No suitable stream found (best/worst).")
        else: raise ValueError("No streams found by Streamlink.")

        # Start the actual processing loop as a task
        processing_task = asyncio.create_task(
            stream_processing_loop(stream_direct_url),
            name="stream_processing_loop"
        )
        stream_active_tasks.add(processing_task)
        await processing_task # Wait for the loop to finish (or be cancelled)

    except Exception as e:
        logger.error(f"MANAGER: Error starting or running stream processing: {e}", exc_info=True)
        stream_is_running = False # Ensure flag is set to false on manager error
    finally:
        logger.info("MANAGER: Stream processing loop has ended.")
        if processing_task in stream_active_tasks:
             stream_active_tasks.remove(processing_task)
        # Ensure flag is false if loop exited unexpectedly
        stream_is_running = False
        # Optional: Add final cleanup here if needed, though shutdown event handles some


async def stream_processing_loop(stream_direct_url: str):
    """The core loop for downloading and scheduling analysis (stream mode)."""
    global stream_is_running, stream_fact_check_buffer, stream_last_fact_check_time, stream_active_tasks

    # --- Method #1: Initial Skip ---
    initial_buffer_duration = 10
    logger.info(f"LOOP: Waiting {initial_buffer_duration} seconds for initial buffer...")
    # Allow cancellation during sleep
    try: await asyncio.sleep(initial_buffer_duration)
    except asyncio.CancelledError: logger.info("LOOP: Initial sleep cancelled."); return

    chunk_duration = 10
    playback_delay = 30
    cycle_target_time = chunk_duration # Aim to finish a cycle in chunk duration

    current_chunk_path_for_analysis = None
    next_chunk_path_downloaded = None
    download_task = None
    analysis_task = None
    chunk_index = 0
    results_buffer = []
    stream_start_time = time.time()
    last_successful_download_time = time.time()

    logger.info("LOOP: Starting main download/analysis cycle.")
    while stream_is_running:
        cycle_start_time = time.time()
        chunk_index += 1
        loop_chunk_output_path = os.path.join(TEMP_DIR, f"stream_chunk_{chunk_index}.mp4")

        # --- 1. Start Downloading Chunk N+1 ---
        logger.debug(f"LOOP: Cycle {chunk_index} starting.")
        if stream_is_running: # Check flag before starting task
            download_task = asyncio.create_task(
                stream_download_chunk(stream_direct_url, loop_chunk_output_path, chunk_duration),
                name=f"download-{chunk_index}"
            )
            stream_active_tasks.add(download_task)
        else: break # Exit if stopped

        # --- 2. Schedule Analysis for Chunk N ---
        if current_chunk_path_for_analysis and stream_is_running:
            logger.debug(f"LOOP: Scheduling analysis for chunk {chunk_index - 1}")
            analysis_task = asyncio.create_task(
                stream_analyze_chunk(current_chunk_path_for_analysis, chunk_index - 1),
                name=f"analyze-{chunk_index-1}"
            )
            stream_active_tasks.add(analysis_task)
            # Add callback to handle analysis result when done
            analysis_task.add_done_callback(
                lambda task: handle_analysis_completion(task, results_buffer)
            )

        # --- 3. Wait for Download of Chunk N+1 ---
        download_success = False
        next_chunk_path_downloaded = None # Reset for this cycle
        if download_task:
            try:
                # Check stream_is_running periodically while waiting
                wait_start = time.time()
                while time.time() - wait_start < 1000: # Generous timeout
                    if not stream_is_running: raise asyncio.CancelledError("Stream stopped during download wait")
                    if download_task.done(): break
                    await asyncio.sleep(0.1) # Short sleep to yield control

                if not download_task.done():
                     logger.warning(f"LOOP: Download task for chunk {chunk_index} taking too long, cancelling.")
                     download_task.cancel()
                     await asyncio.sleep(0.1) # Allow cancellation to propagate
                     raise asyncio.TimeoutError(f"Download timeout for chunk {chunk_index}")

                download_success = await download_task # Get result (True/False) or raise exception
                if download_success:
                    next_chunk_path_downloaded = loop_chunk_output_path
                    last_successful_download_time = time.time()
                else:
                    logger.error(f"LOOP: Download failed reported for chunk {chunk_index}.")
                    # Decide: Stop vs Retry? Retry adds complexity. Let's stop for now.
                    stream_is_running = False; break

            except asyncio.CancelledError:
                logger.info(f"LOOP: Download task for chunk {chunk_index} was cancelled.")
                if stream_is_running: stream_is_running = False # Ensure stop if cancelled externally
                break # Exit loop
            except asyncio.TimeoutError:
                logger.error(f"LOOP: Timeout waiting for download of chunk {chunk_index}. Stopping.")
                stream_is_running = False; break
            except Exception as e:
                logger.error(f"LOOP: Error awaiting download task for chunk {chunk_index}: {e}")
                stream_is_running = False; break
            finally:
                 if download_task in stream_active_tasks: stream_active_tasks.remove(download_task)


        # --- 4. Release Ready Results from Buffer ---
        # This check happens every cycle, releasing results based on time delay
        try:
             await release_buffered_results(results_buffer, stream_start_time, playback_delay, chunk_duration)
        except Exception as e:
             logger.error(f"LOOP: Error releasing buffered results: {e}")


        # --- 5. Prepare for Next Cycle ---
        # Clean up the chunk file that was just analyzed (N-1)
        # We keep chunk N (current_chunk_path_for_analysis) because analysis might still be running
        # We keep chunk N+1 (next_chunk_path_downloaded) because it's needed for the *next* cycle
        chunk_to_delete_index = chunk_index - 6
        if chunk_to_delete_index > 0:
             path_to_delete = os.path.join(TEMP_DIR, f"stream_chunk_{chunk_to_delete_index}.mp4")
             if os.path.exists(path_to_delete):
                 try:
                     os.remove(path_to_delete)
                     logger.debug(f"LOOP: Deleted old chunk {os.path.basename(path_to_delete)}")
                 except Exception as e:
                     logger.warning(f"LOOP: Failed to delete old chunk {path_to_delete}: {e}")

        # The chunk just downloaded (N+1) becomes the one to analyze in the next cycle
        current_chunk_path_for_analysis = next_chunk_path_downloaded

        # --- 6. Cycle Timing ---
        cycle_duration = time.time() - cycle_start_time
        sleep_time = max(0, cycle_target_time - cycle_duration)
        if cycle_duration > cycle_target_time:
            logger.warning(f"LOOP: Cycle {chunk_index} took {cycle_duration:.2f}s (>{cycle_target_time}s).")
        # Check for stalled downloads (if we haven't had a success in a while)
        if time.time() - last_successful_download_time > 3 * cycle_target_time:
             logger.error("LOOP: No successful download in a while. Stopping stream.")
             stream_is_running = False; break

        # Sleep briefly to yield control, but prioritize keeping up if behind
        try:
            await asyncio.sleep(sleep_time * 0.5) # Sleep for only part of the spare time
        except asyncio.CancelledError:
             logger.info("LOOP: Sleep cancelled."); break


    logger.info("LOOP: Exiting stream processing loop.")
    # Final cleanup attempt for remaining tasks if loop exits
    logger.info(f"LOOP: Cancelling {len(stream_active_tasks)} remaining active tasks...")
    for task in list(stream_active_tasks): # Iterate over a copy
         if not task.done():
              task.cancel()
    await asyncio.sleep(0.5) # Allow cancellations
    logger.info("LOOP: Active tasks cancelled.")


def handle_analysis_completion(task: asyncio.Task, results_buffer: List):
    """Callback function to process the result of an analysis task."""
    global stream_fact_check_buffer, stream_last_fact_check_time
    if task.cancelled():
        logger.warning(f"ANALYSIS_CB: Task {task.get_name()} was cancelled.")
        return
    if task.exception():
        logger.error(f"ANALYSIS_CB: Task {task.get_name()} failed: {task.exception()}")
        return

    analysis_results_dict = task.result()
    chunk_index = int(task.get_name().split('-')[-1]) # Extract index from task name

    if analysis_results_dict:
        # Buffer transcription for periodic fact-checking
        transcription = analysis_results_dict.get("transcription", {}).get("english")
        if transcription:
            stream_fact_check_buffer += " " + transcription

        # Add result to buffer, initially without fact checks
        final_result_for_chunk = {
            "video_chunk_url": f"/{TEMP_DIR}/stream_chunk_{chunk_index}.mp4", # Relative URL
            "chunk_index": chunk_index,
            "analysis_timestamp": time.time(),
            "deepfake_analysis": analysis_results_dict.get("deepfake_analysis"),
            "transcription": analysis_results_dict.get("transcription"),
            "fact_check_results": [],
            "fact_check_context_current": False # Flag if FC ran for THIS chunk's callback
        }

        # --- Periodic Fact-Checking ---
        current_time = time.time()
        if current_time - stream_last_fact_check_time >= STREAM_FACT_CHECK_BUFFER_DURATION and stream_fact_check_buffer.strip():
            logger.info(f"ANALYSIS_CB: Running fact check on buffer (length {len(stream_fact_check_buffer)})...")
            try:
                 # Run fact-checking SYNCHRONOUSLY within the callback for simplicity?
                 # Or schedule it as another task? Sync might be okay if FC is fast enough.
                 # Let's try sync first, can change to async task if it blocks too long.
                 fact_check_run_result = fact_checker.check(stream_fact_check_buffer.strip(), num_workers=2)
                 # Store results with this chunk
                 final_result_for_chunk["fact_check_results"] = fact_check_run_result.get("processed_claims", [])
                 final_result_for_chunk["fact_check_context_current"] = True
                 logger.info(f"ANALYSIS_CB: Fact check complete. Found {len(final_result_for_chunk['fact_check_results'])} claims.")
                 stream_fact_check_buffer = "" # Clear buffer
                 stream_last_fact_check_time = current_time
            except Exception as fc_e:
                 logger.error(f"ANALYSIS_CB: Fact checking failed: {fc_e}")
                 final_result_for_chunk["fact_check_results"] = [{"error": str(fc_e)}]
                 final_result_for_chunk["fact_check_context_current"] = True # Indicate check was attempted

        # Insert into buffer sorted by chunk index
        inserted = False
        for i in range(len(results_buffer)):
             if results_buffer[i]["chunk_index"] > chunk_index:
                 results_buffer.insert(i, final_result_for_chunk)
                 inserted = True
                 break
        if not inserted:
            results_buffer.append(final_result_for_chunk)

        logger.debug(f"ANALYSIS_CB: Added result for chunk {chunk_index} to buffer. Buffer size: {len(results_buffer)}")

    else:
        logger.warning(f"ANALYSIS_CB: Task {task.get_name()} returned None result.")

    # Remove self from active tasks (important!)
    if task in stream_active_tasks:
         stream_active_tasks.remove(task)


async def release_buffered_results(results_buffer: List, stream_start_time: float, playback_delay: int, chunk_duration: int):
    """Checks buffer and sends results ready for playback to the WebSocket queue."""
    global stream_result_queue
    elapsed_stream_time = time.time() - stream_start_time
    current_playback_chunk_index = int((elapsed_stream_time - playback_delay) / chunk_duration)

    while results_buffer and results_buffer[0]["chunk_index"] <= current_playback_chunk_index:
        result_to_send = results_buffer.pop(0)
        try:
            # Non-blocking put or short timeout? Let's use blocking for now.
            await stream_result_queue.put(result_to_send)
            logger.info(f"RELEASE: Sent result for chunk {result_to_send['chunk_index']} to WS queue (Queue size: {stream_result_queue.qsize()})")
        except Exception as e:
            logger.error(f"RELEASE: Failed to put result for chunk {result_to_send['chunk_index']} in WS queue: {e}")
            # Should we put it back? Or discard? Discarding might lose data.
            # Put it back at the start maybe?
            results_buffer.insert(0, result_to_send) # Put back if queue fails
            break # Stop trying to release if queue is broken/full


# === FastAPI Endpoints ===

# --- STREAM Endpoints ---
@app.post("/api/stream/analyze")
async def start_stream_analysis(data: dict):
    """Starts the stream analysis background task."""
    global stream_is_running, stream_fact_check_buffer, stream_last_fact_check_time, stream_active_tasks
    url = data.get("url") # Platform URL
    if not url: raise HTTPException(status_code=400, detail="Stream URL is required")

    if stream_is_running:
        logger.warning("Start request received but stream analysis is already running.")
        return {"message": "Stream analysis is already running."}

    logger.info(f"Received start analysis request for: {url}")
    stream_is_running = True
    stream_fact_check_buffer = ""
    stream_last_fact_check_time = time.time()
    # Clear results queue
    while not stream_result_queue.empty():
        try: stream_result_queue.get_nowait(); stream_result_queue.task_done()
        except asyncio.QueueEmpty: break
        except Exception: pass
    # Clear any old tasks (shouldn't be any, but safety)
    for task in list(stream_active_tasks): task.cancel()
    stream_active_tasks.clear()

    # Start the manager task
    asyncio.create_task(stream_processing_manager(url), name="stream_processing_manager")

    return {"message": "Stream analysis starting..."}

@app.websocket("/api/stream/results")
async def stream_results_websocket(websocket: WebSocket):
    """Handles WebSocket connections for sending stream results."""
    await websocket.accept()
    client_id = f"{websocket.client.host}:{websocket.client.port}"
    logger.info(f"WebSocket client connected: {client_id}")
    try:
        while True:
            # Wait for a result indefinitely from the queue
            result = await stream_result_queue.get()
            try:
                await websocket.send_json(result)
                logger.debug(f"Sent chunk {result.get('chunk_index')} data to WS {client_id}")
            except Exception as send_err:
                 logger.error(f"Failed to send message to WS client {client_id}: {send_err}")
                 stream_result_queue.task_done() # Mark done even if send failed for this client
                 break # Disconnect this client
            stream_result_queue.task_done() # Mark item as processed by this client

    except asyncio.CancelledError:
         logger.info(f"WebSocket task cancelled for {client_id}.")
    except Exception as e:
        # Catch potential WebSocket disconnects or other errors
        if "1000 (OK)" in str(e) or "1001 (going away)" in str(e) or "Connection closed" in str(e):
             logger.info(f"WebSocket client {client_id} disconnected normally.")
        else:
             logger.error(f"WebSocket error for client {client_id}: {type(e).__name__} - {e}")
    finally:
        logger.info(f"WebSocket connection closed for {client_id}")
        # Ensure websocket is closed from server-side if loop exits unexpectedly
        try: await websocket.close()
        except: pass


@app.post("/api/stream/stop")
async def stop_stream_analysis():
    """Stops the stream analysis background tasks."""
    global stream_is_running, stream_active_tasks
    if not stream_is_running:
        return {"message": "Stream analysis is not running."}

    logger.info("Received stop analysis request.")
    stream_is_running = False # Signal loops to stop

    # Cancel all tracked stream tasks
    logger.info(f"Attempting to cancel {len(stream_active_tasks)} active stream tasks...")
    cancelled_count = 0
    for task in list(stream_active_tasks): # Iterate copy
        if not task.done():
             task.cancel()
             cancelled_count += 1
    logger.info(f"Cancellation requested for {cancelled_count} tasks.")

    # Give tasks a moment to respond to cancellation
    await asyncio.sleep(1.5) # Increased sleep

    # Clear queue after stopping tasks
    while not stream_result_queue.empty():
         try: stream_result_queue.get_nowait(); stream_result_queue.task_done()
         except asyncio.QueueEmpty: break
         except Exception: pass

    logger.info("Stream analysis stopping sequence complete.")
    return {"message": "Stream analysis stopped"}


# --- UPLOAD MODE ENDPOINTS and HELPERS (Unchanged from original) ---

# Helper to broadcast upload progress
async def broadcast_progress(progress: float):
    # Use list copy for safe iteration if clients disconnect during broadcast
    disconnected_clients = []
    for client in upload_progress_clients:
        try:
            await client.send_text(json.dumps({"progress": progress}))
        except Exception as e:
            logger.warning(f"Failed to send progress to client: {e}. Removing client.")
            disconnected_clients.append(client)

    # Remove disconnected clients
    for client in disconnected_clients:
        if client in upload_progress_clients:
            upload_progress_clients.remove(client)


# Helper to consume Spark results (if Spark is used for uploads)
def consume_spark_results(video_path: str, timeout: int = 60) -> list:
    # This function remains blocking, called within the async endpoint using to_thread if needed
    # Or adjust Spark setup for async consumption if possible
    # consumer = KafkaConsumer(...) # Keep original Kafka consumer logic here
    # results = []
    # ... consume logic ...
    # consumer.close()
    # return results
    logger.warning("Spark result consumption logic is placeholder in this combined script.")
    return [] # Placeholder


@app.post("/api/video/translate")
async def translate_transcription(data: dict):
    # This uses Groq, shared functionality
    transcription = data.get("transcription")
    target_language = data.get("language", "en") # Default target is English

    if not transcription:
        raise HTTPException(status_code=400, detail="Transcription is required")

    try:
        logger.info(f"Translating text to {target_language}: {transcription[:50]}...")
        # Reusing the translate_to_english logic, but adapting the prompt
        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
        payload = {
            "model": "llama3-8b-8192",
            "messages": [
                {"role": "system", "content": f"You are a translator. Translate the following text to {target_language}. Only provide the translated text. Be accurate."},
                {"role": "user", "content": transcription}
            ], "max_tokens": 1024, "temperature": 0.1
        }
        response = requests.post(url, headers=headers, json=payload, timeout=30) # Added timeout
        response.raise_for_status() # Raise HTTP errors

        translation = response.json().get("choices", [{}])[0].get("message", {}).get("content", "[Translation Failed]")
        logger.info(f"Translation result: {translation[:50]}...")
        return {"translation": translation}

    except requests.exceptions.RequestException as req_err:
         logger.error(f"Groq API request error during translation: {req_err}")
         raise HTTPException(status_code=502, detail=f"Translation service API error: {req_err}")
    except Exception as e:
        logger.error(f"Translation error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Translation failed: {str(e)}")

# Add the separate translate_to_english used by stream mode
async def translate_to_english(transcription: str, source_language: str) -> str:
    """Translate text to English using Groq API if not already in English."""
    if source_language == "en" or not transcription:
        return transcription
    if not GROQ_API_KEY:
        logger.warning("GROQ_API_KEY not set, cannot translate to English.")
        return transcription # Fallback

    logger.debug(f"Translating to English from {source_language}: {transcription[:50]}...")
    try:
        # Use aiohttp for async request if possible, or run requests.post in thread
        # Using requests.post in thread for now:
        def do_translate():
            url = "https://api.groq.com/openai/v1/chat/completions"
            headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
            payload = { "model": "llama3-8b-8192",
                "messages": [
                    {"role": "system", "content": "Translate the following text to English. Output only the translation."},
                    {"role": "user", "content": transcription}
                ], "max_tokens": 1024, "temperature": 0.1 }
            response = requests.post(url, headers=headers, json=payload, timeout=20)
            response.raise_for_status()
            return response.json().get("choices", [{}])[0].get("message", {}).get("content", transcription) # Fallback

        translated_text = await asyncio.to_thread(do_translate)
        logger.debug(f"English translation result: {translated_text[:50]}...")
        return translated_text
    except Exception as e:
        logger.error(f"Translation to English failed: {str(e)}")
        return transcription # Fallback to original

@app.websocket("/api/video/progress")
async def video_progress_websocket(websocket: WebSocket):
    """Handles WebSocket connections for sending video upload progress."""
    await websocket.accept()
    client_id = f"{websocket.client.host}:{websocket.client.port}"
    logger.info(f"Upload progress client connected: {client_id}")
    upload_progress_clients.append(websocket)
    try:
        while True:
            # Keep connection open, receive potential messages (optional)
            # If client sends data, handle or ignore: await websocket.receive_text()
            await asyncio.sleep(1) # Keep alive
            # Check if client is still connected (FastAPI handles disconnects,
            # but explicit check can be added if needed)
    except Exception as e:
         # Handle disconnects or errors
        if "1000 (OK)" in str(e) or "1001 (going away)" in str(e):
             logger.info(f"Upload progress client {client_id} disconnected normally.")
        else:
            logger.error(f"Upload progress WebSocket error for {client_id}: {e}")
    finally:
        logger.info(f"Upload progress client disconnected: {client_id}")
        if websocket in upload_progress_clients:
            upload_progress_clients.remove(websocket)


@app.post("/api/video/analyze")
async def analyze_video_upload(file: UploadFile = File(...)):
    """Analyzes an uploaded video file (original logic)."""
    # Use a unique temp path for each upload
    temp_path = os.path.join(TEMP_DIR, f"upload_{int(time.time())}_{file.filename}")
    logger.info(f"Received video upload: {file.filename}, saving to {temp_path}")

    try:
        # Save uploaded file asynchronously
        async with open(temp_path, "wb") as f:
            content = await file.read() # Read full file content
            await f.write(content)
        logger.info(f"Saved uploaded file: {temp_path}, size: {len(content)} bytes")
        await broadcast_progress(0.05) # Initial progress

        # --- Original Analysis Steps for Upload (using Kafka/Spark or direct) ---

        # Option 1: Using Kafka Producer + Spark Consumer (as in original)
        # frame_info = video_producer.process_video(temp_path) # Sends frames to Kafka
        # total_frames = frame_info["total_frames"]
        # logger.info(f"Sent {total_frames} frames to Kafka topic 'video-frames'")
        # await broadcast_progress(0.1)
        # # Consume results from Spark via Kafka
        # spark_results = await asyncio.to_thread(consume_spark_results, temp_path, timeout=120) # Increased timeout
        # await broadcast_progress(0.5)
        # if not spark_results: # Fallback if Spark fails/times out
        #      logger.warning("Spark processing failed or timed out. Falling back to direct analysis.")
        #      # Run direct analysis (similar to original fallback)
        #      transcription, final_score, frames_data, detected_language = await asyncio.to_thread(
        #          video_analyzer.analyze_video, temp_path # video_analyzer.analyze_video needs deepfake detector internally
        #      )
        #      english_transcription = await translate_to_english(transcription, detected_language)
        # else:
        #      # Process Spark results (as in original)
        #      logger.info(f"Received {len(spark_results)} results from Spark.")
        #      frames_data = { # Reconstruct frame data from Spark results
        #          "timestamps": [r.get("timestamp", 0.0) for r in spark_results],
        #          "max_scores": [r.get("deepfake_score", 0.0) for r in spark_results],
        #          "faces_detected": [True] * len(spark_results) # Assuming Spark did face detection or placeholder
        #      }
        #      final_score = max(frames_data["max_scores"]) if frames_data["max_scores"] else 0.0
        #      # Need to get transcription separately if Spark doesn't do it
        #      # Run transcription on the original file (or make Spark do it)
        #      transcription, _, _, detected_language = await asyncio.to_thread( # Run whisper part only
        #             video_analyzer.analyze_video, temp_path
        #      ) # Need to adapt analyze_video or call whisper directly
        #      english_transcription = await translate_to_english(transcription, detected_language)
        #      await broadcast_progress(0.6)


        # Option 2: Direct Analysis (No Kafka/Spark - Simpler, potentially slower for large files)
        logger.info("Performing direct analysis on uploaded video...")
        await broadcast_progress(0.1)
        # video_analyzer.analyze_video includes transcription and deepfake detection now
        # It should return 4 values: transcription, final_score, frames_data, detected_language
        transcription, final_score, frames_data, detected_language = await asyncio.to_thread(
             video_analyzer.analyze_video, temp_path
        )
        logger.info(f"Direct analysis complete. Score: {final_score}, Lang: {detected_language}")
        await broadcast_progress(0.4)
        # Translate transcription if needed
        english_transcription = await translate_to_english(transcription, detected_language)
        logger.info("Transcription translated to English (if necessary).")
        await broadcast_progress(0.5)


        # --- Common Analysis Steps (Fact Check, Text Analysis) ---
        logger.info("Running fact-checking on English transcription...")
        # Use FactChecker instance directly
        fact_check_result = await asyncio.to_thread(fact_checker.check, english_transcription, num_workers=2)
        processed_claims = fact_check_result.get("processed_claims", [])
        logger.info(f"Fact-checking complete. Found {len(processed_claims)} processed claims.")
        await broadcast_progress(0.7)

        logger.info("Running text analysis on English transcription...")
        # Use OptimizedAnalyzer instance directly
        analysis_result = await text_analyzer.analyze_text(english_transcription)
        logger.info("Text analysis complete.")
        await broadcast_progress(0.9)

        # --- Knowledge Graph Update (using FactChecker's KG capability or Analyzer's) ---
        # FactChecker's check method might already update Neo4j if configured.
        # Or, use the Analyzer's KG manager if needed. Let's assume FactChecker handles it.
        # kg_manager = KnowledgeGraphManager() # Not needed if FC does it
        # kg_manager.visualize_graph(...) # Visualization can be generated if needed

        await broadcast_progress(1.0) # Done

        # --- Prepare Response ---
        response = {
            "original_transcription": transcription,
            "detected_language": detected_language,
            "english_transcription": english_transcription,
            "deepfake_final_score": final_score, # Renamed for clarity
            "deepfake_frames_data": frames_data, # Renamed for clarity
            "fact_check_analysis": { # Group fact-check results
                "processed_claims": processed_claims,
                "summary": fact_check_result.get("summary", ""),
                # Add other parts of fact_check_result if needed
            },
            "text_analysis": { # Group text analysis results
                "political_bias": analysis_result.political_bias,
                "emotional_triggers": analysis_result.emotional_triggers,
                "stereotypes": analysis_result.stereotypes,
                "manipulation_score": analysis_result.manipulation_score,
                "entities": analysis_result.entities,
                "locations": analysis_result.locations,
                # Add knowledge graph link/data if generated and needed
                # "knowledge_graph_url": f"/{TEMP_DIR}/knowledge_graph_{os.path.basename(temp_path)}.html"
            }
        }
        logger.info(f"Analysis complete for upload {file.filename}. Sending response.")
        return JSONResponse(content=response)

    except FileNotFoundError:
        logger.error(f"Uploaded file not found after saving: {temp_path}")
        raise HTTPException(status_code=500, detail="Failed to save uploaded file.")
    except ValueError as ve: # E.g., failed to open video
         logger.error(f"Value error during upload analysis: {ve}")
         raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error(f"Error analyzing uploaded video {file.filename}: {str(e)}", exc_info=True)
        # Send final error progress
        await broadcast_progress(-1.0) # Indicate error with negative progress
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")
    finally:
        # Ensure temporary upload file is deleted
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
                logger.info(f"Deleted temporary upload file: {temp_path}")
            except Exception as e_del:
                logger.error(f"Failed to delete temporary upload file {temp_path}: {e_del}")


# Endpoint to serve KG visualization (if generated by upload analysis)
# @app.get("/knowledge_graph/{filename}") async def get_knowledge_graph(...)


# === SHUTDOWN ===
@app.on_event("shutdown")
def shutdown_event():
    global stream_is_running
    logger.info("Application shutdown initiated.")
    stream_is_running = False # Signal stream loop to stop

    # Close Kafka Producer (used by upload)
    try: video_producer.close()
    except Exception as e: logger.error(f"Error closing Kafka Producer: {e}")

    # Stop Spark (used by upload, if active)
    # try: spark_processor.stop()
    # except Exception as e: logger.error(f"Error stopping Spark: {e}")

    # Close Neo4j connection (used by FactChecker)
    try: fact_checker.close_neo4j()
    except Exception as e: logger.error(f"Error closing Neo4j: {e}")

    # Cancel any remaining stream tasks (should be handled by stop endpoint, but safety)
    logger.info(f"Shutdown: Cancelling {len(stream_active_tasks)} potentially remaining tasks...")
    for task in list(stream_active_tasks):
         if not task.done(): task.cancel()

    # Delete remaining temp files
    logger.info(f"Shutdown: Cleaning up temp directory: {TEMP_DIR}")
    if os.path.exists(TEMP_DIR):
        for filename in os.listdir(TEMP_DIR):
            file_path = os.path.join(TEMP_DIR, filename)
            try:
                if os.path.isfile(file_path): os.unlink(file_path)
            except Exception as e: logger.error(f"Error deleting file {file_path} : {e}")

    logger.info("Application shutdown sequence finished.")


# === MAIN GUARD ===
if __name__ == "__main__":
    import uvicorn
    logger.info("Starting Uvicorn server...")
    # Ensure port matches frontend calls and component configurations (e.g., Kafka if external)
    uvicorn.run(app, host="127.0.0.1", port=5001, log_level="info")