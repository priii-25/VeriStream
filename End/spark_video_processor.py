#spark_video_processor.py
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *
import json
from typing import Iterator
import cv2
import numpy as np
from PIL import Image
import io
import base64
from datetime import datetime
import logging
import threading
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SparkVideoProcessor:
    def __init__(self, app_name="VideoAnalysisPipeline"):
        # Update Spark configuration with specific versions
        self.spark = (SparkSession.builder
                     .appName(app_name)
                     .config("spark.sql.streaming.checkpointLocation", "checkpoint")
                     .config("spark.jars.packages", 
                            "org.apache.spark:spark-sql-kafka-0-10_2.13:3.5.0,"
                            "org.apache.kafka:kafka-clients:3.4.0")
                     .config("spark.streaming.stopGracefullyOnShutdown", "true")
                     .config("spark.executor.memory", "4g")
                     .config("spark.driver.memory", "4g")
                     # Add these configurations to ensure proper Kafka integration
                     .config("spark.sql.streaming.kafka.useDeprecatedOffsetFetching", "false")
                     .config("spark.sql.legacy.timeParserPolicy", "LEGACY")
                     .getOrCreate())
        
        # Define schema for video frames
        self.frame_schema = StructType([
            StructField("frame_id", LongType(), True),
            StructField("timestamp", StringType(), True),
            StructField("frame_data", StringType(), True)
        ])

    def create_streaming_df(self, kafka_bootstrap_servers: str, topic: str):
        """Create a streaming DataFrame from Kafka source."""
        return (self.spark
                .readStream
                .format("kafka")
                .option("kafka.bootstrap.servers", kafka_bootstrap_servers)
                .option("subscribe", topic)
                .option("startingOffsets", "latest")
                .option("failOnDataLoss", "false")  # Add this to handle data loss gracefully
                .option("maxOffsetsPerTrigger", "1000")  # Limit batch size
                .load()
                .select(from_json(col("value").cast("string"), self.frame_schema).alias("frame"))
                .select("frame.*"))

    @staticmethod
    def process_frame(frame_data: str) -> np.ndarray:
        """Convert base64 frame data to numpy array."""
        if frame_data is None:
            logger.warning("Received None frame data. Skipping this frame.")
            return np.array([])  # Return an empty array or handle as needed
    
        try:
            img_data = base64.b64decode(frame_data)
            nparr = np.frombuffer(img_data, np.uint8)
            return cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        except Exception as e:
            logger.error(f"Error processing frame data: {e}")
            return np.array([])  # Return an empty array or handle as needed

    def process_batch(self, df, epoch_id):
        """Process a batch of frames using the deepfake detector."""
        # Convert to pandas for easier processing
        pandas_df = df.toPandas()
        
        if not pandas_df.empty:
            # Process frames in parallel using Spark
            frames = pandas_df['frame_data'].apply(self.process_frame).tolist()
            
            # Batch process frames through the detector
            scores = self.detector.predict_batch(frames)
            
            # Create results DataFrame
            results_df = self.spark.createDataFrame([
                (row.frame_id, row.timestamp, score)
                for row, score in zip(pandas_df.itertuples(), scores)
            ], ["frame_id", "timestamp", "deepfake_score"])
            
            # Write results to Kafka
            (results_df.write
             .format("kafka")
             .option("kafka.bootstrap.servers", "localhost:29092")
             .option("topic", "analysis-results")
             .save())

    def start_streaming(self, detector):
        """Start the streaming analysis pipeline."""
        self.detector = detector
        
        streaming_df = self.create_streaming_df("localhost:29092", "video-frames")
        
        query = (streaming_df.writeStream
                .foreachBatch(self.process_batch)
                .outputMode("update")
                .start())
        
        return query

class SparkTranscriptionProcessor:
    def __init__(self, whisper_model):
        self.whisper_model = whisper_model
        
    def process_audio_batch(self, audio_batch: Iterator[bytes]) -> Iterator[str]:
        """Process a batch of audio segments using Whisper."""
        for audio in audio_batch:
            # Convert audio bytes to format whisper expects
            # This is simplified - you'll need to handle proper audio conversion
            result = self.whisper_model.transcribe(audio)
            yield result["text"]

def update_main():
    """Updated main function to use Spark streaming."""
    def start_streaming_analysis(video_path, analyzer, progress_bar):
        # Initialize Spark processor
        spark_processor = SparkVideoProcessor()
        
        # Start video producer in a separate thread
        producer_thread = threading.Thread(
            target=analyzer.producer.send_video,
            args=(video_path,)
        )
        producer_thread.start()
        
        # Start Spark streaming query
        query = spark_processor.start_streaming(analyzer.detector)
        
        try:
            while query.isActive:
                # Update progress bar based on processed data
                progress = query.lastProgress
                if progress:
                    progress_bar.progress(min(progress.numInputRows / 1000, 1.0))
                time.sleep(1)
        except Exception as e:
            logger.error(f"Streaming error: {e}")
        finally:
            query.stop()
            producer_thread.join()
            
        return query.lastProgress