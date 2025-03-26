from fastapi import FastAPI, UploadFile, File, WebSocket, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from video_processor import VideoProducer
from video_analyzer import VideoAnalyzer
from analyzer import OptimizedAnalyzer
from spark_video_processor import SparkVideoProcessor
from optimized_deepfake_detector import OptimizedDeepfakeDetector
from streamlink import Streamlink
from knowledge_graph import KnowledgeGraphManager
import cv2
import numpy as np
import asyncio
import time
import logging
import subprocess
import os
import json
from kafka import KafkaConsumer
import requests
from dotenv import load_dotenv

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

video_producer = VideoProducer()
video_analyzer = VideoAnalyzer()
text_analyzer = OptimizedAnalyzer(use_gpu=True)
spark_processor = SparkVideoProcessor()
session = Streamlink()
is_running = False
result_queue = asyncio.Queue()
progress_clients = []

async def process_stream(stream_url):
    global is_running
    logger.info(f"Starting stream processing for {stream_url}")
    while is_running:
        try:
            streams = session.streams(stream_url)
            if not streams:
                logger.error(f"No streams found for {stream_url}")
                await asyncio.sleep(2)
                continue

            stream = streams.get("best")
            if not stream:
                logger.error("Best stream not available")
                await asyncio.sleep(2)
                continue

            fd = stream.open()
            temp_video = "temp_stream.mp4"
            temp_audio = "temp_audio.wav"

            cmd = ["ffmpeg", "-i", fd.url if hasattr(fd, 'url') else stream.url, "-t", "1", "-c:v", "libx264", "-c:a", "aac", temp_video, "-y"]
            result = subprocess.run(cmd, capture_output=True, text=True)
            fd.close()
            if result.returncode != 0:
                logger.error(f"FFmpeg capture failed: {result.stderr}")
                await asyncio.sleep(2)
                continue

            cap = cv2.VideoCapture(temp_video)
            frames = []
            timestamp = time.time()
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.resize(frame, (640, 480))
                frames.append(frame)
            cap.release()

            score = 0.0
            if frames:
                detector = OptimizedDeepfakeDetector()
                scores = detector.predict_batch(frames)
                score = max(scores) if scores else 0.0

            cmd = ["ffmpeg", "-i", temp_video, "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", temp_audio, "-y"]
            result = subprocess.run(cmd, capture_output=True, text=True)
            transcription = ""
            if result.returncode == 0 and os.path.exists(temp_audio) and os.path.getsize(temp_audio) > 0:
                transcription_result = video_analyzer.whisper_model.transcribe(temp_audio)
                transcription = transcription_result["text"]
            else:
                logger.warning("No audio extracted or extraction failed")
                transcription = "[No audio]"

            text_analysis = await text_analyzer.analyze_text(transcription)
            result = {
                "transcription": transcription,
                "frames": [{"timestamp": timestamp, "score": score}],
                "textAnalysis": {
                    "sentiment": text_analysis.sentiment,
                    "fact_checks": text_analysis.fact_checks or [],
                    "emotional_triggers": text_analysis.emotional_triggers,
                    "stereotypes": text_analysis.stereotypes,
                    "manipulation_score": text_analysis.manipulation_score,
                    "entities": text_analysis.entities
                }
            }
            await result_queue.put(result)

            if os.path.exists(temp_video):
                os.remove(temp_video)
            if os.path.exists(temp_audio):
                os.remove(temp_audio)

            await asyncio.sleep(1)
        except Exception as e:
            logger.error(f"Stream processing error: {str(e)}")
            await asyncio.sleep(2)

@app.post("/api/stream/get-url")
async def get_stream_url(data: dict):
    url = data.get("url")
    if not url:
        raise HTTPException(status_code=400, detail="Stream URL is required")
    try:
        streams = session.streams(url)
        if not streams:
            raise ValueError("No streams available")
        stream = streams.get("best")
        if not stream:
            raise ValueError("Best stream not available")
        return {"hls_url": stream.url}
    except Exception as e:
        logger.error(f"Failed to fetch stream URL: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch stream: {str(e)}")

@app.post("/api/stream/analyze")
async def start_stream_analysis(data: dict):
    global is_running
    url = data.get("url")
    if not url:
        raise HTTPException(status_code=400, detail="Stream URL is required")
    if not is_running:
        is_running = True
        asyncio.create_task(process_stream(url))
        return {"message": "Stream analysis started"}
    return {"message": "Stream analysis already running"}

@app.websocket("/api/stream/results")
async def stream_results(websocket: WebSocket):
    await websocket.accept()
    try:
        while is_running:
            try:
                result = await asyncio.wait_for(result_queue.get(), timeout=2.0)
                await websocket.send_text(json.dumps(result))
            except asyncio.TimeoutError:
                await websocket.send_text(json.dumps({"message": "Processing..."}))
            await asyncio.sleep(0.5)
    except Exception as e:
        logger.error(f"WebSocket error: {str(e)}")
    finally:
        logger.info("WebSocket connection closed")

@app.post("/api/stream/stop")
async def stop_stream_analysis():
    global is_running
    is_running = False
    while not result_queue.empty():
        await result_queue.get()
    return {"message": "Stream analysis stopped"}

@app.websocket("/api/video/progress")
async def video_progress(websocket: WebSocket):
    await websocket.accept()
    progress_clients.append(websocket)
    try:
        while True:
            await asyncio.sleep(1)
    except Exception as e:
        logger.error(f"WebSocket progress error: {str(e)}")
    finally:
        if websocket in progress_clients:
            progress_clients.remove(websocket)

async def broadcast_progress(progress: float):
    for client in progress_clients[:]:
        try:
            await client.send_text(json.dumps({"progress": progress}))
        except Exception as e:
            logger.error(f"Failed to send progress to client: {str(e)}")
            if client in progress_clients:
                progress_clients.remove(client)

@app.post("/api/video/translate")
async def translate_transcription(data: dict, file: UploadFile = File(...)):
    temp_audio = f"temp_audio_{file.filename.split('.')[0]}.wav"
    target_language = data.get("language", "en")
    try:
        with open(temp_audio, "wb") as f:
            content = await file.read()
            f.write(content)

        url = "https://api.groq.com/openai/v1/audio/translations"
        headers = {"Authorization": f"Bearer {GROQ_API_KEY}"}
        with open(temp_audio, "rb") as audio_file:
            files = {
                "file": (temp_audio, audio_file, "audio/wav"),
                "model": (None, "whisper-large-v3"),
                "language": (None, target_language)
            }
            response = requests.post(url, headers=headers, files=files)
        
        if response.status_code != 200:
            raise HTTPException(status_code=500, detail=f"Groq API error: {response.text}")
        
        translation = response.json().get("text", "Translation failed")
        return {"translation": translation}
    finally:
        if os.path.exists(temp_audio):
            os.remove(temp_audio)

@app.post("/api/video/analyze")
async def analyze_video(file: UploadFile = File(...)):
    temp_path = f"temp_{file.filename}"
    try:
        with open(temp_path, "wb") as f:
            await file.seek(0)
            content = await file.read()
            f.write(content)
        
        frame_info = video_producer.process_video(temp_path)
        total_frames = frame_info["total_frames"]
        await broadcast_progress(0.1)
        
        transcription, final_score, frames_data = video_analyzer.analyze_video(temp_path, None)
        await broadcast_progress(0.3)
        
        analysis_result = await text_analyzer.analyze_text(transcription)
        await broadcast_progress(0.6)
        
        spark_results = consume_spark_results(temp_path, timeout=60)
        if not spark_results or len(spark_results) != total_frames:
            logger.warning(f"Expected {total_frames} results, got {len(spark_results)}")
            _, final_score, frames_data = video_analyzer.analyze_video(temp_path, None)
        else:
            frames_data = {
                "timestamps": [r["timestamp"] for r in spark_results],
                "max_scores": [r["deepfake_score"] for r in spark_results],
                "faces_detected": [True] * len(spark_results)
            }
            final_score = max(frames_data["max_scores"]) if frames_data["max_scores"] else 0.0
        await broadcast_progress(1.0)

        # Handle knowledge graph (dictionary to NetworkX)
        kg_manager = KnowledgeGraphManager()
        if isinstance(analysis_result.knowledge_graph, dict):  # If it's a dict from get_graph_data()
            for node, data in analysis_result.knowledge_graph['nodes']:
                kg_manager.graph.add_node(node, **data)
            for edge in analysis_result.knowledge_graph['edges']:
                kg_manager.graph.add_edge(edge[0], edge[1], **edge[2])
        else:
            kg_manager.graph = analysis_result.knowledge_graph  # If it's already a NetworkX graph
        kg_manager.visualize_graph("knowledge_graph.html")

        response = {
            "transcription": transcription,
            "final_score": final_score,
            "frames_data": frames_data,
            "text_analysis": {
                "sentiment": analysis_result.sentiment,
                "fact_checks": analysis_result.fact_checks,
                "emotional_triggers": analysis_result.emotional_triggers,
                "stereotypes": analysis_result.stereotypes,
                "manipulation_score": analysis_result.manipulation_score,
                "entities": analysis_result.entities,
                "locations": analysis_result.locations,
                "knowledge_graph": "/knowledge_graph"
            }
        }
        return JSONResponse(content=response)
    except Exception as e:
        logger.error(f"Analysis error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

@app.get("/knowledge_graph")
async def get_knowledge_graph():
    kg_html = "knowledge_graph.html"
    if os.path.exists(kg_html):
        return FileResponse(kg_html)
    raise HTTPException(status_code=404, detail="Knowledge graph not found")

def consume_spark_results(video_path: str, timeout: int = 60) -> list:
    consumer = KafkaConsumer(
        'analysis-results',
        bootstrap_servers=['localhost:29092'],
        auto_offset_reset='earliest',
        value_deserializer=lambda x: json.loads(x.decode('utf-8')),
        group_id=f"analysis_{video_path}",
        consumer_timeout_ms=timeout * 1000
    )
    results = []
    logger.info(f"Starting to consume results for {video_path}")
    start_time = time.time()
    while time.time() - start_time < timeout:
        message = consumer.poll(timeout_ms=1000)
        for topic_partition, messages in message.items():
            for msg in messages:
                result = msg.value
                if result.get('video_path') == video_path:
                    results.append(result)
        if results:  # Exit early if we have results
            break
        time.sleep(1)
    consumer.close()
    logger.info(f"Consumed {len(results)} results for {video_path}")
    return results

@app.on_event("shutdown")
def shutdown_event():
    global is_running
    is_running = False
    spark_processor.stop()
    logger.info("Application shutdown complete")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=5000, log_level="debug")