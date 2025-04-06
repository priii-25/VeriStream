# backend/main.py
from fastapi import FastAPI, UploadFile, File, WebSocket, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
# --- Keep imports needed for BOTH modes ---
from video_analyzer import VideoAnalyzer # Used by both
from analyzer import OptimizedAnalyzer # Used by both (or parts of it)
from optimized_deepfake_detector import OptimizedDeepfakeDetector # Used directly by stream, video_analyzer uses its own instance for upload path
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
from typing import Dict, Any, List
import aiofiles # <--- Import aiofiles
import aiofiles.os # <--- Import async os functions from aiofiles

# --- Imports specifically for UPLOAD mode ---
from video_processor import VideoProducer # For Kafka in upload (if Option 1 used)
# from spark_video_processor import SparkVideoProcessor # If Spark is used for upload analysis
# from kafka import KafkaConsumer # For consuming Spark results in upload (if Option 1 used) - Comment out if not using Spark

# --- Imports specifically for STREAM mode ---
from streamlink import Streamlink

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
# FactChecker/Analyzer should load their keys internally from .env

logger = logging.getLogger(__name__)
# Consistent logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(threadName)s] - %(message)s',
    handlers=[logging.StreamHandler(), logging.FileHandler('veristream_backend.log')] # Log to console and file
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("shap").setLevel(logging.WARNING)
logging.getLogger("kafka").setLevel(logging.WARNING) # Quieter Kafka logs unless debugging


# Suppress Hugging Face parallelism warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Ensure temp directory exists
TEMP_DIR = "temp_media"
if not os.path.exists(TEMP_DIR):
    os.makedirs(TEMP_DIR)

app = FastAPI()
# Mount the temp directory for serving files
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
    video_analyzer = VideoAnalyzer()
    text_analyzer = OptimizedAnalyzer(use_gpu=True)
    fact_checker = FactChecker()
    stream_deepfake_detector = OptimizedDeepfakeDetector()
    streamlink_session = Streamlink()
    # Instantiate Kafka producer if needed for upload Option 1
    # Ensure Kafka is running if this is used. Handle potential connection errors.
    try:
        video_producer = VideoProducer()
    except Exception as kafka_err:
        logger.error(f"Failed to initialize Kafka Producer: {kafka_err}. Upload via Kafka will fail.")
        video_producer = None # Set to None if unusable

    # spark_processor = SparkVideoProcessor() # Uncomment if Spark processing for uploads is active

    logger.info("Models and components initialized.")
except Exception as e:
     logger.critical(f"FATAL: Failed to initialize core components: {e}", exc_info=True)
     raise RuntimeError(f"Core component initialization failed: {e}") from e


# --- Global State for STREAM mode ---
stream_is_running = False
stream_result_queue = asyncio.Queue()
stream_fact_check_buffer = ""
STREAM_FACT_CHECK_BUFFER_DURATION = 30
stream_last_fact_check_time = 0
stream_active_tasks = set()

# --- Global State for UPLOAD mode ---
upload_progress_clients = []


# === STREAM MODE FUNCTIONS ===

async def stream_download_chunk(stream_url: str, output_path: str, duration: int) -> bool:
    """Downloads AND RE-ENCODES a stream chunk using FFmpeg."""
    task_id = asyncio.current_task().get_name() if asyncio.current_task() else 'download'
    logger.info(f"[{task_id}] Starting download & re-encode: {os.path.basename(output_path)} for {duration}s")
    cmd = [ "ffmpeg", "-hide_banner", "-loglevel", "warning",
        "-i", stream_url, "-t", str(duration),
        "-c:v", "libx264", "-preset", "ultrafast", "-crf", "28", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "128k", "-f", "mp4", output_path, "-y" ]
    process = await asyncio.create_subprocess_exec(
        *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE )
    stdout, stderr = await process.communicate()
    if process.returncode != 0:
        logger.error(f"[{task_id}] FFmpeg download failed for {os.path.basename(output_path)} (Code: {process.returncode}): {stderr.decode()}")
        return False
    else:
        # Use aiofiles for file system checks
        if await aiofiles.os.path.exists(output_path):
            try:
                stat_result = await aiofiles.os.stat(output_path)
                if stat_result.st_size > 1000: # Check > 1KB approx
                     logger.info(f"[{task_id}] Finished download (re-encoded): {os.path.basename(output_path)}")
                     return True
                else:
                     logger.error(f"[{task_id}] FFmpeg finished but output file {os.path.basename(output_path)} is too small.")
                     await aiofiles.os.remove(output_path)
                     return False
            except Exception as stat_err:
                 logger.error(f"[{task_id}] Error checking size of {os.path.basename(output_path)}: {stat_err}")
                 return False
        else:
             logger.error(f"[{task_id}] FFmpeg finished but output file {os.path.basename(output_path)} is missing.")
             return False

async def stream_analyze_chunk(chunk_path: str, chunk_index: int) -> Dict[str, Any] | None:
    """Analyzes a single 10-second video chunk (stream mode)."""
    task_id = asyncio.current_task().get_name() if asyncio.current_task() else f'analyze-{chunk_index}'
    if not await aiofiles.os.path.exists(chunk_path):
        logger.warning(f"[{task_id}] Chunk file not found: {chunk_path}")
        return None

    logger.info(f"[{task_id}] Analyzing chunk: {os.path.basename(chunk_path)}")
    analysis_start_time = time.time()
    analysis_data = {}
    analysis_frame = None
    actual_frame_time = -1.0
    cap = None
    audio_path = os.path.join(TEMP_DIR, f"stream_audio_{chunk_index}.wav")

    try:
        # Video Analysis
        cap = cv2.VideoCapture(chunk_path)
        if not cap.isOpened(): raise ValueError(f"Failed to open chunk {os.path.basename(chunk_path)}")
        target_frame_time_ms = 5 * 1000
        while True:
            ret, frame = cap.read()
            if not ret: break
            current_time_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
            if current_time_ms > 1000:
                if analysis_frame is None and current_time_ms >= target_frame_time_ms:
                    analysis_frame = frame.copy(); actual_frame_time = current_time_ms / 1000.0
                    logger.debug(f"[{task_id}] Selected frame at ~{actual_frame_time:.2f}s")
                    break
        cap.release()

        deepfake_score = 0.0
        if analysis_frame is not None:
            try:
                resized_frame = cv2.resize(analysis_frame, (640, 480))
                scores = await asyncio.to_thread(stream_deepfake_detector.predict_batch, [resized_frame])
                deepfake_score = scores[0] if scores else 0.0
                analysis_data["deepfake_analysis"] = {"timestamp": actual_frame_time, "score": deepfake_score}
                logger.debug(f"[{task_id}] Deepfake score: {deepfake_score:.3f}")
            except Exception as df_err:
                 logger.error(f"[{task_id}] Deepfake prediction failed: {df_err}")
                 analysis_data["deepfake_analysis"] = {"timestamp": actual_frame_time, "score": -1.0}
        else:
            logger.warning(f"[{task_id}] No frame found for deepfake in {os.path.basename(chunk_path)}")
            analysis_data["deepfake_analysis"] = {"timestamp": -1.0, "score": 0.0}

        # Audio Analysis
        transcription_data = { "original": "[Audio Fail]", "detected_language": "unknown", "english": "" }
        cmd_audio = [ "ffmpeg", "-hide_banner", "-loglevel", "error", "-i", chunk_path, "-ss", "1", "-t", "9",
                      "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", audio_path, "-y" ]
        audio_proc = await asyncio.create_subprocess_exec(*cmd_audio, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        _, audio_stderr = await audio_proc.communicate()

        if audio_proc.returncode == 0 and await aiofiles.os.path.exists(audio_path):
             stat_res = await aiofiles.os.stat(audio_path)
             if stat_res.st_size > 44:
                try:
                    logger.debug(f"[{task_id}] Transcribing audio chunk {chunk_index}")
                    trans_result = await asyncio.to_thread(video_analyzer.whisper_model.transcribe, audio_path, fp16=False) # Use base model instance
                    original_text = trans_result["text"].strip(); detected_lang = trans_result["language"]
                    english_text = await translate_to_english(original_text, detected_lang)
                    transcription_data = {"original": original_text, "detected_language": detected_lang, "english": english_text}
                    logger.debug(f"[{task_id}] Transcription done: Lang={detected_lang}, Text={original_text[:30]}...")
                except Exception as whisper_err:
                     logger.error(f"[{task_id}] Whisper failed: {whisper_err}")
                     transcription_data["original"] = "[Transcription Error]"
             else:
                  logger.warning(f"[{task_id}] Audio file empty {os.path.basename(audio_path)}")
                  transcription_data["original"] = "[No Audio Data]"
                  await aiofiles.os.remove(audio_path)
        else:
            logger.warning(f"[{task_id}] Audio extract failed (Code: {audio_proc.returncode}) {os.path.basename(chunk_path)}. Stderr: {audio_stderr.decode()}")
            transcription_data["original"] = "[Audio Extraction Error]"

        analysis_data["transcription"] = transcription_data

    except Exception as e:
         logger.error(f"[{task_id}] Error in chunk analysis {chunk_path}: {e}", exc_info=True)
         return None
    finally:
        if cap and cap.isOpened(): cap.release()
        if await aiofiles.os.path.exists(audio_path):
            try: await aiofiles.os.remove(audio_path)
            except Exception as e_del: logger.warning(f"[{task_id}] Could not remove temp audio {audio_path}: {e_del}")

    logger.info(f"[{task_id}] Chunk {chunk_index} analysis finished in {time.time() - analysis_start_time:.2f}s")
    return analysis_data


async def stream_processing_manager(stream_url_platform: str):
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
        await processing_task
    except Exception as e: logger.error(f"MANAGER: Error: {e}", exc_info=True)
    finally:
        logger.info("MANAGER: Stream processing finishing.")
        if processing_task in stream_active_tasks: stream_active_tasks.remove(processing_task)
        if stream_is_running: logger.warning("MANAGER: Loop exited, stopping stream."); stream_is_running = False


async def stream_processing_loop(stream_direct_url: str):
    global stream_is_running, stream_active_tasks
    initial_buffer_duration = 10
    logger.info(f"LOOP: Waiting {initial_buffer_duration}s...")
    try: await asyncio.sleep(initial_buffer_duration)
    except asyncio.CancelledError: logger.info("LOOP: Initial sleep cancelled."); return

    chunk_duration = 10; playback_delay = 30; cycle_target_time = chunk_duration
    current_chunk_path_for_analysis = None; next_chunk_path_downloaded = None
    download_task = None; analysis_task = None
    chunk_index = 0; results_buffer = []
    stream_start_time = time.time(); last_successful_download_time = time.time()

    logger.info("LOOP: Starting main cycle.")
    while stream_is_running:
        cycle_start_time = time.time(); chunk_index += 1
        loop_chunk_output_path = os.path.join(TEMP_DIR, f"stream_chunk_{chunk_index}.mp4")
        logger.debug(f"LOOP: Cycle {chunk_index}.")

        # --- 1. Start Download N+1 ---
        if not stream_is_running: break
        download_task = asyncio.create_task( stream_download_chunk(stream_direct_url, loop_chunk_output_path, chunk_duration), name=f"download-{chunk_index}" )
        stream_active_tasks.add(download_task)

        # --- 2. Schedule Analysis N ---
        if current_chunk_path_for_analysis and stream_is_running:
            logger.debug(f"LOOP: Scheduling analysis chunk {chunk_index - 1}")
            if analysis_task and not analysis_task.done():
                 logger.warning(f"LOOP: Prev analysis {analysis_task.get_name()} unfinished, cancelling."); analysis_task.cancel()
                 if analysis_task in stream_active_tasks: stream_active_tasks.remove(analysis_task)
            analysis_task = asyncio.create_task( stream_analyze_chunk(current_chunk_path_for_analysis, chunk_index - 1), name=f"analyze-{chunk_index-1}" )
            stream_active_tasks.add(analysis_task)
            analysis_task.add_done_callback(lambda task: handle_analysis_completion(task, results_buffer))

        # --- 3. Wait for Download N+1 ---
        download_success = False; next_chunk_path_downloaded = None
        if download_task:
            try:
                wait_start = time.time()
                while time.time() - wait_start < cycle_target_time * 2.5:
                    if not stream_is_running: raise asyncio.CancelledError("Stream stopped")
                    if download_task.done(): break
                    await asyncio.sleep(0.1)
                if not download_task.done(): download_task.cancel(); await asyncio.sleep(0.1); raise asyncio.TimeoutError(f"Dl timeout {chunk_index}")
                download_success = await download_task
                if download_success: next_chunk_path_downloaded = loop_chunk_output_path; last_successful_download_time = time.time()
                else: raise RuntimeError(f"Dl failed {chunk_index}")
            except (asyncio.CancelledError, asyncio.TimeoutError, RuntimeError) as e:
                logger.error(f"LOOP: Download {download_task.get_name()} failed/stopped: {type(e).__name__} - {e}. Stopping."); stream_is_running = False; break
            except Exception as e: logger.error(f"LOOP: Error await dl {download_task.get_name()}: {e}", exc_info=True); stream_is_running = False; break
            finally:
                 if download_task in stream_active_tasks: stream_active_tasks.remove(download_task)

        # --- 4. Release Results ---
        try: await release_buffered_results(results_buffer, stream_start_time, playback_delay, chunk_duration)
        except Exception as e: logger.error(f"LOOP: Error releasing results: {e}")

        # --- 5. Prep Next Cycle & Cleanup ---
        current_chunk_path_for_analysis = next_chunk_path_downloaded
        chunk_to_delete_index = chunk_index - 6 # Keep more chunks
        if chunk_to_delete_index > 0:
             path_to_delete = os.path.join(TEMP_DIR, f"stream_chunk_{chunk_to_delete_index}.mp4")
             if await aiofiles.os.path.exists(path_to_delete):
                 try: await aiofiles.os.remove(path_to_delete); logger.debug(f"LOOP: Deleted {os.path.basename(path_to_delete)}")
                 except Exception as e_del: logger.warning(f"LOOP: Failed delete {path_to_delete}: {e_del}")

        # --- 6. Timing ---
        cycle_duration = time.time() - cycle_start_time
        sleep_time = max(0, cycle_target_time - cycle_duration)
        if cycle_duration > cycle_target_time: logger.warning(f"LOOP: Cycle {chunk_index} took {cycle_duration:.2f}s.")
        if time.time() - last_successful_download_time > cycle_target_time * 4: # Increased tolerance
             logger.error("LOOP: No success download recently. Stopping."); stream_is_running = False; break
        try: await asyncio.sleep(sleep_time * 0.5) # Shorter sleep if behind
        except asyncio.CancelledError: logger.info("LOOP: Sleep cancelled."); break

    logger.info("LOOP: Exiting.")
    if analysis_task and not analysis_task.done(): logger.info(f"LOOP: Cancelling task {analysis_task.get_name()}"); analysis_task.cancel()


def handle_analysis_completion(task: asyncio.Task, results_buffer: List):
    global stream_fact_check_buffer, stream_last_fact_check_time, fact_checker, stream_active_tasks
    analysis_results_dict = None; chunk_index = -1
    try:
        chunk_index = int(task.get_name().split('-')[-1])
        if task.cancelled(): logger.warning(f"ANALYSIS_CB: Task {task.get_name()} cancelled."); return
        if task.exception(): logger.error(f"ANALYSIS_CB: Task {task.get_name()} failed: {task.exception()}", exc_info=task.exception()); return
        analysis_results_dict = task.result()
        if analysis_results_dict:
            logger.info(f"ANALYSIS_CB: Processing result chunk {chunk_index}")
            transcription = analysis_results_dict.get("transcription", {}).get("english")
            if transcription: stream_fact_check_buffer += " " + transcription
            final_result = { "video_chunk_url": f"/{TEMP_DIR}/stream_chunk_{chunk_index}.mp4",
                "chunk_index": chunk_index, "analysis_timestamp": time.time(),
                "deepfake_analysis": analysis_results_dict.get("deepfake_analysis"),
                "transcription": analysis_results_dict.get("transcription"),
                "fact_check_results": [], "fact_check_context_current": False }
            # Periodic Fact-Checking (Sync call)
            current_time = time.time()
            run_fc = (current_time - stream_last_fact_check_time >= STREAM_FACT_CHECK_BUFFER_DURATION and stream_fact_check_buffer.strip())
            if run_fc:
                logger.info(f"ANALYSIS_CB: Running fact check (buffer {len(stream_fact_check_buffer)})...")
                buffer_copy = stream_fact_check_buffer; stream_fact_check_buffer = ""; stream_last_fact_check_time = current_time
                try:
                     fc_result = fact_checker.check(buffer_copy.strip(), num_workers=2) # Blocking
                     final_result["fact_check_results"] = fc_result.get("processed_claims", [])
                     final_result["fact_check_context_current"] = True
                     logger.info(f"ANALYSIS_CB: Fact check done. Claims: {len(final_result['fact_check_results'])}.")
                except Exception as fc_e: logger.error(f"ANALYSIS_CB: Fact check failed: {fc_e}", exc_info=True); final_result["fact_check_results"] = [{"error": str(fc_e)}]; final_result["fact_check_context_current"] = True
            # Insert sorted
            ins_idx = next((i for i, r in enumerate(results_buffer) if r["chunk_index"] > chunk_index), -1)
            if ins_idx != -1: results_buffer.insert(ins_idx, final_result)
            else: results_buffer.append(final_result)
            logger.debug(f"ANALYSIS_CB: Added chunk {chunk_index}. Buffer size: {len(results_buffer)}")
        else: logger.warning(f"ANALYSIS_CB: Task {task.get_name()} returned None.")
    except Exception as cb_err: logger.error(f"ANALYSIS_CB: Error task {task.get_name()}: {cb_err}", exc_info=True)
    finally:
        if task in stream_active_tasks: stream_active_tasks.remove(task)


async def release_buffered_results(results_buffer: List, stream_start_time: float, playback_delay: int, chunk_duration: int):
    global stream_result_queue
    if not results_buffer: return
    elapsed_time = time.time() - stream_start_time
    ready_chunk_index = int(max(0, elapsed_time - playback_delay) / chunk_duration) + 1
    while results_buffer and results_buffer[0]["chunk_index"] <= ready_chunk_index:
        result = results_buffer.pop(0)
        try:
            await asyncio.wait_for(stream_result_queue.put(result), timeout=2.0)
            logger.info(f"RELEASE: Sent chunk {result['chunk_index']} (QSize: {stream_result_queue.qsize()})")
        except asyncio.TimeoutError: logger.error(f"RELEASE: Timeout put chunk {result['chunk_index']}. Re-inserting."); results_buffer.insert(0, result); break
        except Exception as e: logger.error(f"RELEASE: Failed put chunk {result['chunk_index']}: {e}"); results_buffer.insert(0, result); break


# === FastAPI Endpoints ===
# --- STREAM Endpoints ---
@app.post("/api/stream/analyze")
async def start_stream_analysis(data: dict):
    global stream_is_running, stream_active_tasks, stream_fact_check_buffer, stream_last_fact_check_time
    url = data.get("url")
    if not url: raise HTTPException(status_code=400, detail="Stream URL required")
    if stream_is_running: return {"message": "Already running."}
    logger.info(f"Start analysis request: {url}")
    stream_is_running = True; stream_fact_check_buffer = ""; stream_last_fact_check_time = time.time()
    while not stream_result_queue.empty(): # Clear queue
        try: stream_result_queue.get_nowait(); stream_result_queue.task_done()
        except asyncio.QueueEmpty: break
    for task in list(stream_active_tasks): task.cancel()
    stream_active_tasks.clear()
    asyncio.create_task(stream_processing_manager(url), name="stream_processing_manager")
    logger.info("Stream manager task created.")
    return {"message": "Stream analysis starting..."}

@app.websocket("/api/stream/results")
async def stream_results_websocket(websocket: WebSocket):
    await websocket.accept(); client_id = f"{websocket.client.host}:{websocket.client.port}"
    logger.info(f"WS client connected: {client_id}")
    try:
        while True:
            result = await stream_result_queue.get()
            try: await websocket.send_json(result)
            except Exception as send_err: logger.error(f"WS Send Error {client_id}: {send_err}"); stream_result_queue.task_done(); break
            stream_result_queue.task_done()
    except asyncio.CancelledError: logger.info(f"WS task cancelled {client_id}.")
    except Exception as e: logger.info(f"WS client {client_id} disconnected: {e}")
    finally: logger.info(f"WS connection closed {client_id}"); await websocket.close()

@app.post("/api/stream/stop")
async def stop_stream_analysis():
    global stream_is_running, stream_active_tasks
    if not stream_is_running: return {"message": "Not running."}
    logger.info("Stop analysis request.")
    stream_is_running = False
    logger.info(f"Cancelling {len(stream_active_tasks)} stream tasks...")
    tasks_to_wait = [task for task in list(stream_active_tasks) if not task.done()]
    for task in tasks_to_wait: task.cancel()
    if tasks_to_wait:
        try: await asyncio.wait_for(asyncio.gather(*tasks_to_wait, return_exceptions=True), timeout=5.0)
        except asyncio.TimeoutError: logger.warning("Timeout waiting for tasks cancel.")
    stream_active_tasks.clear()
    while not stream_result_queue.empty(): # Clear queue
        try: stream_result_queue.get_nowait(); stream_result_queue.task_done()
        except asyncio.QueueEmpty: break
    logger.info("Stream analysis stopped.")
    return {"message": "Stream analysis stopped"}


# === UPLOAD MODE ENDPOINTS and HELPERS (Using aiofiles) ===

async def broadcast_progress(progress: float):
    disconnected_clients = []
    for client in list(upload_progress_clients):
        try: await client.send_text(json.dumps({"progress": progress}))
        except Exception as e: logger.warning(f"Progress send fail: {e}. Removing."); disconnected_clients.append(client)
    for client in disconnected_clients:
        if client in upload_progress_clients: upload_progress_clients.remove(client)

def consume_spark_results(video_path: str, timeout: int = 60) -> list:
    logger.warning("Spark result consumption logic is placeholder.")
    return []

@app.post("/api/video/translate")
async def translate_transcription_endpoint(data: dict):
    transcription = data.get("transcription"); target_language = data.get("language", "en")
    if not transcription: raise HTTPException(status_code=400, detail="Transcription required")
    try:
        translated_text = await translate_to_language(transcription, target_language)
        return {"translation": translated_text}
    except Exception as e: logger.error(f"Translation endpoint error: {e}", exc_info=True); raise HTTPException(status_code=500, detail=f"Translation failed: {str(e)}")

async def translate_to_language(text: str, target_language: str) -> str:
    if not text or not GROQ_API_KEY: return text
    logger.debug(f"Translating to {target_language}...")
    try:
        def do_translate():
            url = "https://api.groq.com/openai/v1/chat/completions"; headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
            payload = { "model": "llama3-8b-8192", "messages": [{"role": "system", "content": f"Translate the following text to {target_language}. Output only the translation."}, {"role": "user", "content": text}], "max_tokens": 1024, "temperature": 0.1 }
            response = requests.post(url, headers=headers, json=payload, timeout=25); response.raise_for_status()
            result = response.json().get("choices", [{}])[0].get("message", {}).get("content", text); return result.strip().strip('"').strip("'")
        translated = await asyncio.to_thread(do_translate)
        logger.debug(f"Translation result: {translated[:50]}...")
        return translated
    except Exception as e: logger.error(f"Translation to {target_language} failed: {e}"); return text

async def translate_to_english(transcription: str, source_language: str) -> str:
    if source_language == "en" or not transcription: return transcription
    return await translate_to_language(transcription, "English")

@app.websocket("/api/video/progress")
async def video_progress_websocket(websocket: WebSocket):
    await websocket.accept(); client_id = f"{websocket.client.host}:{websocket.client.port}"
    logger.info(f"Upload progress client connected: {client_id}"); upload_progress_clients.append(websocket)
    try:
        while True: await asyncio.sleep(3600)
    except Exception: pass
    finally:
        logger.info(f"Upload progress client disconnected: {client_id}")
        if websocket in upload_progress_clients: upload_progress_clients.remove(websocket)

@app.post("/api/video/analyze")
async def analyze_video_upload(file: UploadFile = File(...)):
    temp_path = os.path.join(TEMP_DIR, f"upload_{int(time.time())}_{file.filename}")
    logger.info(f"Upload: {file.filename} -> {temp_path}")
    try:
        # Async save using aiofiles
        content = await file.read()
        async with aiofiles.open(temp_path, "wb") as f: await f.write(content)
        logger.info(f"Upload saved: {temp_path}, size: {len(content)} bytes")
        await broadcast_progress(0.05)

        # Direct Analysis Path
        logger.info(f"Starting direct analysis: {os.path.basename(temp_path)}")
        await broadcast_progress(0.1)
        analysis_tuple = await asyncio.to_thread(video_analyzer.analyze_video, temp_path)
        transcription, final_score, frames_data, detected_language = analysis_tuple
        logger.info(f"Direct video analysis done. Score: {final_score:.3f}, Lang: {detected_language}")
        await broadcast_progress(0.4)
        english_transcription = await translate_to_english(transcription, detected_language)
        await broadcast_progress(0.5)
        logger.info("Running fact-check...")
        fact_check_result = await asyncio.to_thread(fact_checker.check, english_transcription, num_workers=2)
        await broadcast_progress(0.7)
        logger.info("Running text analysis...")
        text_analysis_result = await text_analyzer.analyze_text(english_transcription)
        await broadcast_progress(0.9)
        await broadcast_progress(1.0)

        response = { "original_transcription": transcription, "detected_language": detected_language,
                     "english_transcription": english_transcription, "final_score": final_score, "frames_data": frames_data,
                     "text_analysis": { "political_bias": text_analysis_result.political_bias,
                                        "emotional_triggers": text_analysis_result.emotional_triggers,
                                        "stereotypes": text_analysis_result.stereotypes,
                                        "manipulation_score": text_analysis_result.manipulation_score,
                                        "entities": text_analysis_result.entities,
                                        "locations": text_analysis_result.locations,
                                        "fact_check_result": fact_check_result } }
        logger.info(f"Analysis complete for {file.filename}.")
        return JSONResponse(content=response)
    except Exception as e:
        logger.error(f"Error analyzing upload {file.filename}: {e}", exc_info=True)
        await broadcast_progress(-1.0)
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")
    finally:
        if 'temp_path' in locals() and await aiofiles.os.path.exists(temp_path):
            try: await aiofiles.os.remove(temp_path); logger.info(f"Deleted temp upload: {temp_path}")
            except Exception as e_del: logger.error(f"Failed delete {temp_path}: {e_del}")


# === SHUTDOWN ===
@app.on_event("shutdown")
async def shutdown_event():
    global stream_is_running, stream_active_tasks
    logger.info("Application shutdown initiated.")
    stream_is_running = False
    if stream_active_tasks:
        logger.info(f"Shutdown: Cancelling {len(stream_active_tasks)} stream tasks...")
        for task in list(stream_active_tasks): task.cancel()
        try: await asyncio.wait_for(asyncio.gather(*stream_active_tasks, return_exceptions=True), timeout=3.0)
        except asyncio.TimeoutError: logger.warning("Timeout waiting stream tasks cancel.")
        stream_active_tasks.clear()
    # Close Kafka Producer
    if video_producer:
        try: logger.info("Closing Kafka producer..."); await asyncio.to_thread(video_producer.close); logger.info("Kafka producer closed.")
        except Exception as e: logger.error(f"Error closing Kafka Producer: {e}")
    # Close Neo4j
    try: logger.info("Closing Neo4j connection..."); await asyncio.to_thread(fact_checker.close_neo4j); logger.info("Neo4j connection closed.")
    except Exception as e: logger.error(f"Error closing Neo4j: {e}")
    # Async cleanup of temp dir
    logger.info(f"Shutdown: Cleaning up temp directory: {TEMP_DIR}")
    try:
        if await aiofiles.os.path.isdir(TEMP_DIR):
            filenames = await asyncio.to_thread(os.listdir, TEMP_DIR)
            tasks = []
            for filename in filenames:
                file_path = os.path.join(TEMP_DIR, filename)
                # Schedule deletion task for files
                if await aiofiles.os.path.isfile(file_path):
                     tasks.append(asyncio.create_task(aiofiles.os.remove(file_path), name=f"delete-{filename}"))
            if tasks:
                 await asyncio.gather(*tasks, return_exceptions=True) # Run deletions concurrently
                 logger.info("Temp directory cleanup finished.")
            else:
                 logger.info("Temp directory empty, no cleanup needed.")
    except Exception as e_clean: logger.error(f"Error during temp dir cleanup: {e_clean}")
    logger.info("Application shutdown sequence finished.")


# === MAIN GUARD ===
if __name__ == "__main__":
    import uvicorn
    logger.info("Starting Uvicorn server on http://127.0.0.1:5001")
    uvicorn.run("main:app", host="127.0.0.1", port=5001, log_level="info", reload=True) # Added reload=True for dev