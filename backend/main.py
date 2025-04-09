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
# --- Configure Logging ---
logging.basicConfig(
    level=logging.INFO, # Set default to INFO
    format='%(asctime)s - %(levelname)s - [%(name)s] - %(message)s', # Use name for clarity
    handlers=[logging.StreamHandler(), logging.FileHandler('veristream_backend.log')]
)
# Silence overly verbose libraries unless debugging
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("shap").setLevel(logging.WARNING)
logging.getLogger("kafka").setLevel(logging.WARNING)
logging.getLogger("PIL").setLevel(logging.WARNING)
logging.getLogger("matplotlib").setLevel(logging.WARNING)
# --- Suppress Neo4j INFO/WARNING Notifications ---
logging.getLogger("neo4j.notifications").setLevel(logging.ERROR) # Only show errors

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
    stream_deepfake_detector = OptimizedDeepfakeDetector()
    video_analyzer = VideoAnalyzer()
    text_analyzer = OptimizedAnalyzer(use_gpu=True) # Or False if no GPU
    fact_checker = FactChecker()
    streamlink_session = Streamlink()
    video_producer = None # Initialize as None

    logger.info("Models and components initialized.")
except Exception as e:
     logger.critical(f"FATAL: Failed to initialize core components: {e}", exc_info=True)
     raise RuntimeError(f"Core component initialization failed: {e}") from e

# --- Global State for STREAM mode ---
stream_is_running = False
stream_result_queue = asyncio.Queue()
stream_fact_check_buffer = ""
STREAM_FACT_CHECK_BUFFER_DURATION = 30 # Run fact check every 30 seconds of accumulated text
stream_last_fact_check_time = 0
stream_active_tasks = set()
STREAM_HEATMAP_THRESHOLD = 0.7

# --- NEW Global State for Background Fact-Checking ---
latest_fact_check_results: Optional[Dict] = None # Stores the latest completed result dict {processed_claims:..., summary:...}
fact_check_lock = asyncio.Lock() # Although we'll peek for now, lock is good practice if modifying state later
background_fact_check_task: Optional[asyncio.Task] = None # Track the running background task

# --- Global State for UPLOAD mode ---
upload_progress_clients = []


# === STREAM MODE FUNCTIONS ===

async def stream_download_chunk(stream_url: str, output_path: str, duration: int) -> bool:
    """Downloads AND RE-ENCODES a stream chunk using FFmpeg."""
    task_id = asyncio.current_task().get_name() if asyncio.current_task() else 'download'
    logger.info(f"[{task_id}] Starting download & RE-ENCODE: {os.path.basename(output_path)} for {duration}s") # Explicitly mention re-encode

    # --- Use ONLY the re-encoding command for reliability ---
    cmd = [ "ffmpeg", "-hide_banner", "-loglevel", "warning",
        "-i", stream_url,
        "-t", str(duration), # Limit duration
        # Video Encoding settings (adjust preset/crf for quality/speed balance)
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "26", "-pix_fmt", "yuv420p",
        # Audio Encoding settings
        "-c:a", "aac", "-b:a", "128k",
        # Output format and force overwrite
        "-f", "mp4", output_path, "-y" ]

    process = await asyncio.create_subprocess_exec(
        *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE )
    stdout, stderr = await process.communicate()

    if process.returncode != 0:
        logger.error(f"[{task_id}] FFmpeg re-encode failed for {os.path.basename(output_path)} (Code: {process.returncode}): {stderr.decode()}")
        # Attempt to delete potentially corrupt partial file
        if await aiofiles.os.path.exists(output_path):
            try: await aiofiles.os.remove(output_path)
            except Exception as e_del: logger.warning(f"Failed to remove potentially corrupt file {output_path}: {e_del}")
        return False
    else:
        # --- File check remains the same ---
        try:
            if await aiofiles.os.path.exists(output_path):
                stat_result = await aiofiles.os.stat(output_path)
                if stat_result.st_size > 1000: # Check > 1KB approx
                     logger.info(f"[{task_id}] Finished download (re-encoded): {os.path.basename(output_path)}")
                     return True
                else:
                     logger.error(f"[{task_id}] FFmpeg finished but output file {os.path.basename(output_path)} is too small (likely corrupt).")
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
        if cap.isOpened(): cap.release()

        # --- Deepfake Detection & Heatmap Generation ---
        deepfake_score = 0.0
        heatmap_url = None
        if analysis_frame is not None:
            try:
                scores = await asyncio.to_thread(stream_deepfake_detector.predict_batch, [analysis_frame])
                deepfake_score = scores[0] if scores else 0.0
                logger.debug(f"[{task_id}] Deepfake score: {deepfake_score:.3f}")

                if deepfake_score > STREAM_HEATMAP_THRESHOLD:
                    logger.info(f"[{task_id}] Score {deepfake_score:.3f} > {STREAM_HEATMAP_THRESHOLD}, generating heatmap...")
                    heatmap = await asyncio.to_thread(stream_deepfake_detector.get_attention_maps, analysis_frame)
                    if heatmap is not None:
                        overlay = await asyncio.to_thread(stream_deepfake_detector.create_overlay, analysis_frame, heatmap)
                        if overlay is not None:
                            overlay_filename = f"{base_chunk_filename}_overlay.jpg"
                            overlay_path_abs = os.path.join(TEMP_DIR, overlay_filename)
                            # Save using OpenCV in the thread (blocking but usually fast)
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
        transcription_data = { "original": "[Audio Fail]", "detected_language": "unknown", "english": "" }
        cmd_audio = [ "ffmpeg", "-hide_banner", "-loglevel", "error", "-i", chunk_path,
                      "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", audio_path, "-y" ]
        audio_proc = await asyncio.create_subprocess_exec(*cmd_audio, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        _, audio_stderr = await audio_proc.communicate()

        if audio_proc.returncode == 0:
            if await aiofiles.os.path.exists(audio_path):
                try:
                    stat_res = await aiofiles.os.stat(audio_path)
                    if stat_res.st_size > 44:
                        logger.debug(f"[{task_id}] Transcribing audio chunk {chunk_index}")
                        # Run Whisper in thread
                        trans_result = await asyncio.to_thread(video_analyzer.whisper_model.transcribe, audio_path, fp16=False)
                        original_text = trans_result["text"].strip()
                        detected_lang = trans_result["language"]
                        # Translate if necessary (in thread via helper)
                        english_text = await translate_to_english(original_text, detected_lang)
                        transcription_data = {"original": original_text, "detected_language": detected_lang, "english": english_text}
                        logger.debug(f"[{task_id}] Transcription done: Lang={detected_lang}, Text={original_text[:30]}...")
                    else:
                        logger.warning(f"[{task_id}] Audio file exists but is empty: {os.path.basename(audio_path)}")
                        transcription_data["original"] = "[No Audio Data]"
                        await aiofiles.os.remove(audio_path)
                except Exception as audio_err:
                    logger.error(f"[{task_id}] Audio processing/transcription failed: {audio_err}", exc_info=True)
                    transcription_data["original"] = "[Transcription Error]"
            else:
                logger.error(f"[{task_id}] Audio extract ffmpeg OK, but file missing: {os.path.basename(audio_path)}")
                transcription_data["original"] = "[Audio File Missing]"
        else:
            logger.warning(f"[{task_id}] Audio extract failed (Code: {audio_proc.returncode}) {os.path.basename(chunk_path)}. Stderr: {audio_stderr.decode()}")
            transcription_data["original"] = "[Audio Extraction Error]"

        analysis_data["transcription"] = transcription_data

    except Exception as e:
         logger.error(f"[{task_id}] Error during chunk analysis {chunk_path}: {e}", exc_info=True)
         analysis_data["deepfake_analysis"] = {"timestamp": -1.0, "score": -1.0, "heatmap_url": None}
         analysis_data["transcription"] = {"original": "[Analysis Failed]", "detected_language": "error", "english": ""}
    finally:
        if cap and cap.isOpened(): cap.release()
        if await aiofiles.os.path.exists(audio_path):
            try: await aiofiles.os.remove(audio_path)
            except Exception as e_del: logger.warning(f"[{task_id}] Could not remove temp audio {audio_path}: {e_del}")

    logger.info(f"[{task_id}] Chunk {chunk_index} analysis finished in {time.time() - analysis_start_time:.2f}s")
    return analysis_data

# --- NEW Function to run fact-checking in background ---
async def run_fact_check_in_background(buffer_to_check: str, checker: FactChecker):
    """Runs fact_checker.check in a background thread and stores the result."""
    global latest_fact_check_results, fact_check_lock # Use global state
    task_name = asyncio.current_task().get_name() if asyncio.current_task() else 'fact_check_bg'
    logger.info(f"[{task_name}] Starting background fact check for text length {len(buffer_to_check)}...")
    start_fc_time = time.time()
    results = None
    try:
        # Run the potentially blocking fact-checker in a thread managed by asyncio
        results = await asyncio.to_thread(checker.check, buffer_to_check, num_workers=2) # Use the passed checker instance
        duration = time.time() - start_fc_time
        logger.info(f"[{task_name}] Background fact check completed in {duration:.2f}s. Found {len(results.get('processed_claims', []))} claims.")
    except Exception as e:
        logger.error(f"[{task_name}] Background fact check execution failed: {e}", exc_info=True)
        # Store an error marker or None
        results = {"error": f"Background Fact Check Failed: {e}", "processed_claims": [], "summary": "Error during fact check."}
    finally:
        # Safely update the global variable with the latest results
        # Using lock here is ideal, but peeking might be okay if writes are infrequent
        # async with fact_check_lock:
        latest_fact_check_results = results # Store the full result dict (simple assignment for now)
        logger.debug(f"[{task_name}] Updated latest_fact_check_results.")

# Modified handle_analysis_completion function
def handle_analysis_completion(task: asyncio.Task, results_buffer: List):
    """Processes completed analysis tasks, handles transcription buffering,
       triggers background fact-checking, and includes latest results."""
    # --- Use global variables ---
    global stream_fact_check_buffer, stream_last_fact_check_time, fact_checker
    global stream_active_tasks, latest_fact_check_results # Removed lock for now, using peek
    global background_fact_check_task

    analysis_results_dict = None
    chunk_index = -1
    task_name = task.get_name()

    try:
        # --- Existing logic to get task result ---
        try: chunk_index = int(task_name.split('-')[-1])
        except (IndexError, ValueError): logger.error(f"ANALYSIS_CB: Could not parse chunk index from task name '{task_name}'"); return

        if task.cancelled():
            logger.warning(f"ANALYSIS_CB: Task {task_name} was cancelled."); return

        exc = task.exception()
        if exc:
            logger.error(f"ANALYSIS_CB: Task {task_name} failed with exception: {exc}", exc_info=exc)
            return

        analysis_results_dict = task.result()

        if analysis_results_dict:
            logger.info(f"ANALYSIS_CB: Processing analysis result for chunk {chunk_index}")

            # --- Add Transcription to Buffer ---
            transcription = analysis_results_dict.get("transcription", {}).get("english", "").strip()
            if transcription and transcription != "[Analysis Pending]" and not transcription.startswith("["):
                stream_fact_check_buffer += " " + transcription

            # --- Check if time to trigger a NEW background fact-check ---
            current_time = time.time()
            should_run_new_fact_check = (
                stream_fact_check_buffer.strip() and
                (current_time - stream_last_fact_check_time >= STREAM_FACT_CHECK_BUFFER_DURATION)
            )

            if should_run_new_fact_check:
                 # Prevent starting a new check if one is already running
                 if background_fact_check_task and not background_fact_check_task.done():
                     logger.info(f"ANALYSIS_CB: Skipping new fact check trigger; previous task '{background_fact_check_task.get_name()}' still running.")
                 else:
                    logger.info(f"ANALYSIS_CB: Triggering background fact check on buffered text (length {len(stream_fact_check_buffer)})...")
                    buffer_to_check = stream_fact_check_buffer.strip()
                    # Reset global buffer and timer *before* starting the task
                    stream_fact_check_buffer = ""
                    stream_last_fact_check_time = current_time

                    # Start the fact check in the background - DO NOT AWAIT
                    background_fact_check_task = asyncio.create_task(
                        run_fact_check_in_background(buffer_to_check, fact_checker), # Pass checker instance
                        name=f"bg_fact_check_triggered_by_{chunk_index}"
                    )
                    # Do NOT add to stream_active_tasks unless essential for shutdown cancellation

            # --- Prepare Result for *This* Chunk ---
            current_fc_results_to_send = []
            current_fc_summary = ""
            fact_check_updated_now = False

            # Peek at the latest completed results (no lock for simplicity, slight risk)
            if latest_fact_check_results is not None:
                 logger.debug(f"ANALYSIS_CB: Found completed fact check results to include for chunk {chunk_index}.")
                 current_fc_results_to_send = latest_fact_check_results.get("processed_claims", [])
                 current_fc_summary = latest_fact_check_results.get("summary", "")
                 fact_check_updated_now = True
                 # Clear it *after* copying
                 latest_fact_check_results = None
            # else: logger.debug(f"ANALYSIS_CB: No new fact check results available for chunk {chunk_index}.")


            final_result = {
                "video_chunk_url": f"/{TEMP_DIR}/stream_chunk_{chunk_index}.mp4",
                "chunk_index": chunk_index,
                "analysis_timestamp": time.time(),
                "deepfake_analysis": analysis_results_dict.get("deepfake_analysis"),
                "transcription": analysis_results_dict.get("transcription"),
                # --- Include the latest *completed* fact check results ---
                "fact_check_results": current_fc_results_to_send,
                "fact_check_summary": current_fc_summary, # Send summary too
                "fact_check_context_current": fact_check_updated_now # Flag if FC results are fresh in *this* message
            }

            # --- Insert into results buffer (existing logic) ---
            ins_idx = 0
            while ins_idx < len(results_buffer) and results_buffer[ins_idx]["chunk_index"] < chunk_index:
                ins_idx += 1
            results_buffer.insert(ins_idx, final_result)

            logger.debug(f"ANALYSIS_CB: Added chunk {chunk_index} result. Buffer size: {len(results_buffer)}")

        else:
            logger.warning(f"ANALYSIS_CB: Analysis task {task_name} returned None result.")

    except Exception as cb_err:
        logger.error(f"ANALYSIS_CB: Error in callback for task {task_name}: {cb_err}", exc_info=True)
    finally:
        # Ensure *analysis* task is removed from active set
        if task in stream_active_tasks:
            stream_active_tasks.remove(task)

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
    playback_delay = chunk_duration * 1.5 # Slightly reduced delay
    cycle_target_time = chunk_duration # Target time for one loop iteration

    logger.info(f"LOOP: Waiting {initial_buffer_duration}s for initial stream buffer...")
    try:
        await asyncio.sleep(initial_buffer_duration)
    except asyncio.CancelledError:
        logger.info("LOOP: Initial sleep cancelled.")
        return # Exit if cancelled during initial wait

    current_chunk_path_for_analysis = None
    next_chunk_path_downloaded = None
    download_task = None
    analysis_task = None
    chunk_index = 0
    results_buffer = []
    stream_start_time = time.time()
    last_successful_download_time = time.time()
    max_consecutive_failures = 3
    consecutive_failures = 0

    logger.info("LOOP: Starting main processing cycle.")
    while stream_is_running:
        cycle_start_time = time.time()
        chunk_index += 1
        loop_chunk_output_path = os.path.join(TEMP_DIR, f"stream_chunk_{chunk_index}.mp4")
        logger.debug(f"LOOP: Cycle {chunk_index} starting.")

        # --- 1. Start Download Task for Chunk N+1 ---
        if not stream_is_running: break
        if download_task and not download_task.done():
            logger.warning(f"LOOP: Previous download {download_task.get_name()} still running? Cancelling.")
            download_task.cancel()
            try: await asyncio.sleep(0.1)
            except asyncio.CancelledError: pass
            if download_task in stream_active_tasks: stream_active_tasks.remove(download_task)

        download_task = asyncio.create_task(
            stream_download_chunk(stream_direct_url, loop_chunk_output_path, chunk_duration),
            name=f"download-{chunk_index}"
        )
        stream_active_tasks.add(download_task)
        logger.debug(f"LOOP: Started download task {download_task.get_name()}.")

        # --- 2. Schedule Analysis Task for Chunk N ---
        if current_chunk_path_for_analysis and stream_is_running:
            logger.debug(f"LOOP: Scheduling analysis for previous chunk {chunk_index - 1} ({os.path.basename(current_chunk_path_for_analysis)}).")
            if analysis_task and not analysis_task.done():
                 logger.warning(f"LOOP: Previous analysis {analysis_task.get_name()} unfinished, cancelling.")
                 analysis_task.cancel()
                 try: await asyncio.sleep(0.1)
                 except asyncio.CancelledError: pass
                 if analysis_task in stream_active_tasks: stream_active_tasks.remove(analysis_task)

            analysis_task = asyncio.create_task(
                 stream_analyze_chunk(current_chunk_path_for_analysis, chunk_index - 1),
                 name=f"analyze-{chunk_index-1}"
             )
            stream_active_tasks.add(analysis_task)
            # Use lambda to pass results_buffer to the callback
            analysis_task.add_done_callback(
                lambda task: handle_analysis_completion(task, results_buffer)
            )
            logger.debug(f"LOOP: Scheduled analysis task {analysis_task.get_name()}.")
        elif not current_chunk_path_for_analysis and chunk_index > 1:
             logger.warning(f"LOOP: No chunk path available to schedule analysis for chunk {chunk_index - 1}.")

        # --- 3. Wait for Download Task N+1 to Complete ---
        download_success = False
        next_chunk_path_downloaded = None
        if download_task:
            try:
                wait_start = time.time()
                timeout_duration = cycle_target_time * 2.5 # e.g., 25 seconds
                logger.debug(f"LOOP: Waiting for download {download_task.get_name()} (timeout: {timeout_duration:.1f}s)")

                while time.time() - wait_start < timeout_duration:
                     if not stream_is_running:
                         if not download_task.done(): download_task.cancel()
                         raise asyncio.CancelledError("Stream stopped during download wait")
                     if download_task.done(): break
                     await asyncio.sleep(0.1)

                if not download_task.done():
                     download_task.cancel()
                     try: await asyncio.sleep(0.1)
                     except asyncio.CancelledError: pass
                     raise asyncio.TimeoutError(f"Download task {download_task.get_name()} timed out after {timeout_duration:.1f}s")

                download_success = download_task.result()

                if download_success:
                    next_chunk_path_downloaded = loop_chunk_output_path
                    last_successful_download_time = time.time()
                    consecutive_failures = 0
                    logger.debug(f"LOOP: Download {download_task.get_name()} successful.")
                else:
                    raise RuntimeError(f"Download task {download_task.get_name()} reported failure.")

            except (asyncio.CancelledError, asyncio.TimeoutError, RuntimeError) as e:
                logger.error(f"LOOP: Download {download_task.get_name()} failed or stopped: {type(e).__name__} - {e}.")
                consecutive_failures += 1
                if not isinstance(e, asyncio.CancelledError) or stream_is_running:
                     stream_is_running = False # Stop loop on critical errors unless already stopping
                break # Exit loop on critical download failures
            except Exception as e:
                logger.error(f"LOOP: Unexpected error awaiting download {download_task.get_name()}: {e}", exc_info=True)
                consecutive_failures += 1
                stream_is_running = False
                break
            finally:
                 if download_task in stream_active_tasks:
                     stream_active_tasks.remove(download_task)

        # --- 4. Release Ready Results from Buffer ---
        if stream_is_running:
            try:
                await release_buffered_results(results_buffer, stream_start_time, playback_delay, chunk_duration)
            except Exception as e:
                logger.error(f"LOOP: Error releasing results: {e}")

        # --- 5. Prep for Next Cycle & Cleanup Old Files ---
        current_chunk_path_for_analysis = next_chunk_path_downloaded
        chunk_to_delete_index = chunk_index - (HISTORY_LIMIT + CLEANUP_BUFFER)

        if chunk_to_delete_index > 0:
             logger.debug(f"LOOP: Checking for cleanup eligibility for index <= {chunk_to_delete_index}")
             base_path_to_delete = os.path.join(TEMP_DIR, f"stream_chunk_{chunk_to_delete_index}")
             path_to_delete_mp4 = f"{base_path_to_delete}.mp4"
             path_to_delete_overlay = f"{base_path_to_delete}_overlay.jpg"

             async def delete_file(f_path, f_type):
                 try:
                     if await aiofiles.os.path.exists(f_path):
                         await aiofiles.os.remove(f_path)
                         logger.info(f"LOOP: Deleted old {f_type} file: {os.path.basename(f_path)}")
                 except Exception as e_del:
                     logger.warning(f"LOOP: Failed to delete old {f_type} file {f_path}: {e_del}")

             asyncio.create_task(delete_file(path_to_delete_mp4, "chunk"))
             asyncio.create_task(delete_file(path_to_delete_overlay, "overlay"))

        # --- 6. Timing & Health Check ---
        cycle_duration = time.time() - cycle_start_time
        sleep_time = max(0, cycle_target_time - cycle_duration)

        if cycle_duration > cycle_target_time * 1.1:
            logger.warning(f"LOOP: Cycle {chunk_index} duration ({cycle_duration:.2f}s) exceeded target ({cycle_target_time:.1f}s).")
            sleep_time = 0

        if time.time() - last_successful_download_time > cycle_target_time * (max_consecutive_failures + 1):
             logger.error(f"LOOP: No successful download in ~{cycle_target_time * (max_consecutive_failures + 1):.0f}s. Stopping stream.")
             stream_is_running = False
             break

        if consecutive_failures >= max_consecutive_failures:
             logger.error(f"LOOP: Stopping stream after {consecutive_failures} consecutive failures.")
             stream_is_running = False
             break

        if sleep_time > 0 and stream_is_running:
            try:
                await asyncio.sleep(sleep_time)
            except asyncio.CancelledError:
                logger.info("LOOP: Cycle sleep cancelled.")
                break

    logger.info("LOOP: Exiting main processing loop.")

    if analysis_task and not analysis_task.done():
        logger.info(f"LOOP: Cancelling final analysis task {analysis_task.get_name()} on exit.")
        analysis_task.cancel()
        try: await asyncio.wait_for(analysis_task, timeout=1.0)
        except (asyncio.TimeoutError, asyncio.CancelledError): pass
        except Exception as e: logger.warning(f"LOOP: Error during final analysis task cancellation ({analysis_task.get_name()}): {e}")
        finally:
            if analysis_task in stream_active_tasks: stream_active_tasks.remove(analysis_task)

    logger.info("LOOP: Finished cleanup on exit.")


async def release_buffered_results(results_buffer: List, stream_start_time: float, playback_delay: int, chunk_duration: int):
    # (No changes needed here)
    global stream_result_queue
    if not results_buffer: return

    elapsed_time = time.time() - stream_start_time
    ready_chunk_index_threshold = int(max(0, elapsed_time - playback_delay) / chunk_duration)

    while results_buffer and results_buffer[0]["chunk_index"] <= ready_chunk_index_threshold:
        result_to_release = results_buffer.pop(0)
        try:
            await asyncio.wait_for(stream_result_queue.put(result_to_release), timeout=2.0)
            logger.info(f"RELEASE: Sent chunk {result_to_release['chunk_index']} results to WS queue (QSize: {stream_result_queue.qsize()})")
        except asyncio.TimeoutError:
            logger.error(f"RELEASE: Timeout putting chunk {result_to_release['chunk_index']} onto WS queue. Re-inserting into buffer.")
            results_buffer.insert(0, result_to_release)
            break
        except Exception as e:
            logger.error(f"RELEASE: Failed to put chunk {result_to_release['chunk_index']} onto WS queue: {e}")
            results_buffer.insert(0, result_to_release)
            break


# === FastAPI Endpoints ===

# --- STREAM Endpoints ---
@app.post("/api/stream/analyze")
async def start_stream_analysis(data: dict):
    global stream_is_running, stream_active_tasks, stream_fact_check_buffer, stream_last_fact_check_time
    global stream_result_queue, latest_fact_check_results, background_fact_check_task
    url = data.get("url")
    if not url: raise HTTPException(status_code=400, detail="Stream URL required")
    if stream_is_running:
        logger.warning("Start request received but stream is already running.")
        return {"message": "Stream analysis is already running."}

    logger.info(f"Start analysis request received for URL: {url}")

    # --- Reset State ---
    stream_is_running = True
    stream_fact_check_buffer = ""
    stream_last_fact_check_time = time.time()
    latest_fact_check_results = None # Clear previous results
    if background_fact_check_task and not background_fact_check_task.done():
        background_fact_check_task.cancel() # Cancel any lingering bg task
        background_fact_check_task = None

    while not stream_result_queue.empty():
        try: stream_result_queue.get_nowait(); stream_result_queue.task_done()
        except asyncio.QueueEmpty: break
        except Exception as e: logger.warning(f"Error clearing result queue item: {e}")

    if stream_active_tasks:
         logger.warning(f"Found {len(stream_active_tasks)} active tasks before start. Cancelling them.")
         tasks_to_cancel = list(stream_active_tasks)
         for task in tasks_to_cancel: task.cancel()
         try:
             await asyncio.wait_for(asyncio.gather(*tasks_to_cancel, return_exceptions=True), timeout=1.0)
         except asyncio.TimeoutError: logger.warning("Timeout waiting for old tasks to cancel during start.")
         stream_active_tasks.clear()

    # --- Start Manager ---
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
            queue_get_task = asyncio.create_task(stream_result_queue.get())
            # Wait for either a result or a client message (like close)
            receive_task = asyncio.create_task(websocket.receive_text())

            done, pending = await asyncio.wait(
                {queue_get_task, receive_task},
                return_when=asyncio.FIRST_COMPLETED
            )

            # Prioritize processing the queue result if ready
            if queue_get_task in done:
                 if receive_task in pending: receive_task.cancel() # Cancel receive if queue was first
                 result = queue_get_task.result()
                 try:
                     await websocket.send_json(result)
                     stream_result_queue.task_done()
                 except Exception as send_err:
                      logger.error(f"WS Send Error to {client_id}: {send_err}. Client likely disconnected.")
                      stream_result_queue.task_done() # Mark done anyway to avoid blocking queue
                      break
            elif receive_task in done:
                 # Client sent message or disconnected
                 if queue_get_task in pending: queue_get_task.cancel() # Cancel queue get
                 try:
                     # Process potential message (usually just indicates closure)
                     message = receive_task.result()
                     logger.info(f"WS client {client_id} sent message: {message}") # Or just log disconnect
                 except Exception as ws_err:
                      # This likely catches WebSocketDisconnect
                      logger.info(f"WS client {client_id} disconnected: {type(ws_err).__name__}")
                 break # Exit loop on client disconnect/message
            else:
                 # Should not happen with FIRST_COMPLETED
                 logger.warning("WS wait returned unexpected state.")
                 if queue_get_task in pending: queue_get_task.cancel()
                 if receive_task in pending: receive_task.cancel()
                 break


    except asyncio.CancelledError:
        logger.info(f"WebSocket task for {client_id} cancelled.")
        if queue_get_task and not queue_get_task.done(): queue_get_task.cancel()
    except Exception as e:
        logger.info(f"WebSocket client {client_id} error: {type(e).__name__} - {e}")
        if queue_get_task and not queue_get_task.done(): queue_get_task.cancel()
    finally:
        logger.info(f"WebSocket connection closing for {client_id}")
        # Cancel any pending tasks associated with this specific connection if necessary
        if queue_get_task and not queue_get_task.done(): queue_get_task.cancel()
        # Let FastAPI handle the actual closing


@app.post("/api/stream/stop")
async def stop_stream_analysis():
    global stream_is_running, stream_active_tasks, stream_result_queue
    global background_fact_check_task # Ensure this is included

    if not stream_is_running:
        logger.info("Stop request received but stream is not running.")
        return {"message": "Stream analysis is not running."}

    logger.info("Stop analysis request received.")
    stream_is_running = False # Signal loops and tasks to stop

    # --- Cancel Background Fact-Check Task ---
    if background_fact_check_task and not background_fact_check_task.done():
        logger.info("Stop: Cancelling background fact-check task...")
        background_fact_check_task.cancel()
        # Add it to the main list for unified cancellation waiting
        if background_fact_check_task not in stream_active_tasks:
            stream_active_tasks.add(background_fact_check_task)
        background_fact_check_task = None # Clear reference

    # --- Cancel Other Active Tasks ---
    logger.info(f"Cancelling {len(stream_active_tasks)} active stream tasks...")
    tasks_to_cancel = list(stream_active_tasks)
    for task in tasks_to_cancel:
        if not task.done(): task.cancel()

    if tasks_to_cancel:
        try:
            await asyncio.wait_for(asyncio.gather(*tasks_to_cancel, return_exceptions=True), timeout=5.0)
            logger.info("Active stream tasks cancelled.")
        except asyncio.TimeoutError:
            logger.warning("Timeout waiting for stream tasks to complete cancellation.")
        except Exception as e:
             logger.error(f"Error during task cancellation gathering: {e}")

    stream_active_tasks.clear()

    # --- Clear Result Queue ---
    logger.info("Clearing stream result queue...")
    cleared_count = 0
    while not stream_result_queue.empty():
        try:
            stream_result_queue.get_nowait()
            stream_result_queue.task_done()
            cleared_count += 1
        except asyncio.QueueEmpty: break
        except Exception as e: logger.warning(f"Error clearing stream result queue item: {e}"); break
    logger.info(f"Cleared {cleared_count} items from result queue.")

    logger.info("Stream analysis stopped successfully.")
    return {"message": "Stream analysis stopped"}


# === UPLOAD MODE ENDPOINTS and HELPERS ===

async def broadcast_progress(progress: float):
    # (No changes needed here)
    disconnected_clients = []
    message = json.dumps({"progress": progress})
    for client in list(upload_progress_clients):
        try:
            await client.send_text(message)
        except Exception as e:
            logger.warning(f"Progress send failed ({type(e).__name__}). Removing client.")
            disconnected_clients.append(client)
    for client in disconnected_clients:
        if client in upload_progress_clients:
            try: upload_progress_clients.remove(client)
            except ValueError: pass

async def translate_to_language(text: str, target_language: str) -> str:
    if not text or not GROQ_API_KEY: return text
    if target_language.lower() == "english": return text # Simpler check

    logger.debug(f"Translating to {target_language} using Groq...")
    try:
        def do_translate():
            url = "https://api.groq.com/openai/v1/chat/completions"
            headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
            payload = { "model": "llama3-8b-8192",
                "messages": [
                    {"role": "system", "content": f"Translate the following text accurately to {target_language}. Output *only* the translated text, without any introductory phrases, explanations, or quotation marks."},
                    {"role": "user", "content": text} ],
                "max_tokens": 2048, "temperature": 0.1 }
            try:
                response = requests.post(url, headers=headers, json=payload, timeout=30)
                response.raise_for_status()
                result = response.json()
                translated_content = result.get("choices", [{}])[0].get("message", {}).get("content", "")
                return translated_content.strip().strip('"').strip("'").strip()
            except requests.exceptions.Timeout: logger.error("Translation timed out."); return f"[Translation Timeout]"
            except requests.exceptions.RequestException as req_err: logger.error(f"Translation API failed: {req_err}"); return f"[Translation API Error]"
            except Exception as json_err: logger.error(f"Translation parse failed: {json_err}"); return f"[Translation Parse Error]"

        translated = await asyncio.to_thread(do_translate)
        logger.debug(f"Translation result (truncated): {translated[:50]}...")
        return translated
    except Exception as e: logger.error(f"Translation system error: {e}", exc_info=True); return f"[Translation System Error]"

async def translate_to_english(transcription: str, source_language: str) -> str:
    if source_language.lower().startswith("en") or not transcription: return transcription
    logger.info(f"Translating from '{source_language}' to English...")
    return await translate_to_language(transcription, "English")


@app.post("/api/video/translate")
async def translate_transcription_endpoint(data: dict):
    # (No changes needed here)
    transcription = data.get("transcription")
    target_language = data.get("language", "en")
    if not transcription: raise HTTPException(status_code=400, detail="Transcription text required")
    if not target_language: raise HTTPException(status_code=400, detail="Target language required")
    try:
        translated_text = await translate_to_language(transcription, target_language)
        return {"translation": translated_text}
    except Exception as e: logger.error(f"Translation endpoint error: {e}", exc_info=True); raise HTTPException(status_code=500, detail=f"Translation failed: {str(e)}")

@app.websocket("/api/video/progress")
async def video_progress_websocket(websocket: WebSocket):
    # (No changes needed here)
    await websocket.accept()
    client_id = f"{websocket.client.host}:{websocket.client.port}"
    logger.info(f"Upload progress client connected: {client_id}")
    upload_progress_clients.append(websocket)
    try:
        while True: await websocket.receive_text()
    except Exception as e: logger.info(f"Upload progress client {client_id} disconnected: {type(e).__name__}")
    finally:
        logger.info(f"Upload progress client connection closed: {client_id}")
        if websocket in upload_progress_clients:
            try: upload_progress_clients.remove(websocket)
            except ValueError: pass

@app.post("/api/video/analyze")
async def analyze_video_upload(file: UploadFile = File(...)):
    # (No changes needed here - already uses VideoAnalyzer correctly)
    timestamp = int(time.time())
    safe_filename = "".join(c for c in file.filename if c.isalnum() or c in ('_', '-')).rstrip()
    temp_path = os.path.join(TEMP_DIR, f"upload_{timestamp}_{safe_filename}")
    logger.info(f"Upload request: {file.filename} -> {temp_path}")
    await broadcast_progress(0.0)
    try:
        async with aiofiles.open(temp_path, "wb") as f:
            while content := await file.read(1024 * 1024): await f.write(content)
        if not await aiofiles.os.path.exists(temp_path): raise IOError(f"Failed to save uploaded file to {temp_path}")
        stat_res = await aiofiles.os.stat(temp_path)
        logger.info(f"Upload saved: {temp_path}, size: {stat_res.st_size} bytes")
        await broadcast_progress(0.05)

        logger.info(f"Starting direct analysis (incl. heatmaps): {os.path.basename(temp_path)}")
        await broadcast_progress(0.1)

        async def upload_progress_update(analyzer_progress: float):
             backend_progress = 0.1 + analyzer_progress * 0.85
             await broadcast_progress(backend_progress)

        # Use the existing VideoAnalyzer which calls the detector internally
        analysis_tuple = await asyncio.to_thread( video_analyzer.analyze_video, temp_path, upload_progress_update )
        transcription, final_score, frames_data, detected_language = analysis_tuple

        logger.info(f"Direct video analysis done. Score: {final_score:.3f}, Lang: {detected_language}")

        english_transcription = await translate_to_english(transcription, detected_language)
        await broadcast_progress(0.96)

        logger.info("Running fact-check...")
        # Use the main fact_checker instance for consistency
        fact_check_result = await asyncio.to_thread(fact_checker.check, english_transcription, num_workers=2)
        await broadcast_progress(0.97)

        logger.info("Running text analysis...")
        # Use the main text_analyzer instance
        text_analysis_result = await text_analyzer.analyze_text(english_transcription) # Assuming this is internally async or thread-safe
        await broadcast_progress(0.99)

        response = {
            "original_transcription": transcription,
            "detected_language": detected_language,
            "english_transcription": english_transcription,
            "final_score": final_score,
            "frames_data": frames_data,
            "text_analysis": {
                "political_bias": text_analysis_result.political_bias,
                "emotional_triggers": text_analysis_result.emotional_triggers,
                "stereotypes": text_analysis_result.stereotypes,
                "manipulation_score": text_analysis_result.manipulation_score,
                "entities": text_analysis_result.entities,
                "locations": text_analysis_result.locations,
                "fact_check_result": fact_check_result # Include full fact check results here
            }
        }
        await broadcast_progress(1.0)
        logger.info(f"Analysis complete for {file.filename}.")
        return JSONResponse(content=response)

    except FileNotFoundError as fnf_err: logger.error(f"File not found: {fnf_err}"); await broadcast_progress(-1.0); raise HTTPException(status_code=404, detail=str(fnf_err))
    except IOError as io_err: logger.error(f"File save/read error: {io_err}"); await broadcast_progress(-1.0); raise HTTPException(status_code=500, detail=f"File handling error: {io_err}")
    except Exception as e: logger.error(f"Error analyzing upload {file.filename}: {e}", exc_info=True); await broadcast_progress(-1.0); raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")
    finally:
        if 'temp_path' in locals() and await aiofiles.os.path.exists(temp_path):
            try: await aiofiles.os.remove(temp_path); logger.info(f"Deleted temp upload file: {temp_path}")
            except Exception as e_del: logger.error(f"Failed to delete temp upload file {temp_path}: {e_del}")


# === SHUTDOWN ===
@app.on_event("shutdown")
async def shutdown_event():
    # --- Use global variables ---
    global stream_is_running, stream_active_tasks, video_producer, fact_checker
    global background_fact_check_task # Added

    logger.info("Application shutdown initiated.")

    # --- Stop Stream Gracefully ---
    stream_is_running = False # Signal stream loops to stop

    # --- Also cancel the background fact-check task if it's running ---
    if background_fact_check_task and not background_fact_check_task.done():
        logger.info("Shutdown: Cancelling background fact-check task...")
        background_fact_check_task.cancel()
        # Add it to the list to wait for, if desired (short timeout)
        if background_fact_check_task not in stream_active_tasks: # Avoid duplicates
             stream_active_tasks.add(background_fact_check_task)
        # No need to clear reference here, task cancellation handles it

    # --- Existing Task Cancellation Logic ---
    if stream_active_tasks:
        logger.info(f"Shutdown: Cancelling {len(stream_active_tasks)} active stream/background tasks...")
        tasks_to_cancel = list(stream_active_tasks) # Copy before iterating
        for task in tasks_to_cancel:
            if not task.done(): task.cancel()
        try:
            await asyncio.wait_for(asyncio.gather(*tasks_to_cancel, return_exceptions=True), timeout=3.0)
            logger.info("Stream/background tasks cancellation complete.")
        except asyncio.TimeoutError: logger.warning("Timeout waiting for tasks to cancel during shutdown.")
        except Exception as e: logger.error(f"Error during task cancellation gathering: {e}")
        stream_active_tasks.clear()

    # --- Close External Connections (Kafka, Neo4j) ---
    if video_producer:
        try: logger.info("Closing Kafka producer..."); await asyncio.to_thread(video_producer.close); logger.info("Kafka producer closed.")
        except Exception as e: logger.error(f"Error closing Kafka Producer: {e}")

    try:
        logger.info("Closing Neo4j connection via FactChecker...")
        await asyncio.to_thread(fact_checker.close_neo4j)
        logger.info("Neo4j connection closed.")
    except Exception as e: logger.error(f"Error closing Neo4j connection: {e}", exc_info=True)

    # --- Async Cleanup of Temp Directory ---
    logger.info(f"Shutdown: Cleaning up temp directory: {TEMP_DIR}")
    cleanup_tasks = []
    try:
        if await aiofiles.os.path.isdir(TEMP_DIR):
            # --- FIX: Await listdir before iterating ---
            filenames = await aiofiles.os.listdir(TEMP_DIR)
            for filename in filenames:
                file_path = os.path.join(TEMP_DIR, filename)
                try:
                     if await aiofiles.os.path.isfile(file_path):
                          cleanup_tasks.append(
                              asyncio.create_task(aiofiles.os.remove(file_path), name=f"delete-{filename}")
                          )
                except Exception as stat_err: logger.warning(f"Error stating file {file_path} during cleanup: {stat_err}")

            if cleanup_tasks:
                 logger.info(f"Attempting to delete {len(cleanup_tasks)} files from {TEMP_DIR}...")
                 results = await asyncio.gather(*cleanup_tasks, return_exceptions=True)
                 success_count = 0
                 for i, result in enumerate(results):
                     task_name = cleanup_tasks[i].get_name()
                     if isinstance(result, Exception): logger.error(f"Failed to delete file during cleanup ({task_name}): {result}")
                     else: success_count += 1
                 logger.info(f"Temp directory cleanup finished. Deleted {success_count}/{len(cleanup_tasks)} files.")
            else: logger.info("Temp directory empty or contained no files, no cleanup needed.")
    except FileNotFoundError: logger.info(f"Temp directory {TEMP_DIR} not found, skipping cleanup.")
    except Exception as e_clean: logger.error(f"Error during temp directory cleanup: {e_clean}", exc_info=True)

    logger.info("Application shutdown sequence finished.")


# === MAIN GUARD ===
if __name__ == "__main__":
    import uvicorn
    logger.info("Starting Uvicorn server on http://127.0.0.1:5001")
    uvicorn.run(
        "main:app",
        host="127.0.0.1",
        port=5001,
        log_config=None, # Use our configured logger
        reload=False # Disable reload for stable testing/production
        # reload=True # Use ONLY for development
    )