# backend/spark_video_processor.py
import os
from pyspark.sql import SparkSession
from optimized_deepfake_detector import OptimizedDeepfakeDetector
import base64
import cv2
import numpy as np
import json
import logging
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, FloatType, DoubleType

logger = logging.getLogger('spark_video_processor')

class SparkVideoProcessor:
    def __init__(self):
        """Initialize Spark session with Kafka support."""
        try:
            os.environ['SPARK_HOME'] = '/Users/kartik/Desktop/vs/VeriStream/spark-3.5.3-bin-hadoop3-scala2.13'
            os.environ['PYSPARK_PYTHON'] = '/Users/kartik/Desktop/vs/VeriStream/env/bin/python'

            self.spark = SparkSession.builder \
                .appName("VideoAnalysis") \
                .config("spark.master", "local[*]") \
                .config("spark.jars.packages", "org.apache.spark:spark-sql-kafka-0-10_2.13:3.5.0") \
                .config("spark.sql.streaming.checkpointLocation", "/tmp/spark_checkpoint") \
                .config("spark.default.parallelism", "8") \
                .config("spark.sql.shuffle.partitions", "8") \
                .getOrCreate()

            self.detector = OptimizedDeepfakeDetector()
            logger.info("SparkVideoProcessor initialized with Spark session and Kafka support")
        except Exception as e:
            logger.error(f"Failed to initialize SparkVideoProcessor: {str(e)}")
            raise

    def process_frame(self, frame_data: str) -> np.ndarray:
        """Decode a base64-encoded frame into a NumPy array."""
        try:
            img_data = base64.b64decode(frame_data)
            nparr = np.frombuffer(img_data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if frame is None:
                raise ValueError("Failed to decode frame")
            return frame
        except Exception as e:
            logger.error(f"Error decoding frame: {str(e)}")
            return None

    def process_batch(self, df, epoch_id):
        """Process a batch of frames from Kafka."""
        try:
            if df.isEmpty():
                logger.info(f"Batch {epoch_id}: No data received from Kafka, skipping processing")
                return

            # Define schema as SQL string for from_json
            schema_str = "frame_id INT, timestamp DOUBLE, video_path STRING, frame_data STRING"
            df = df.selectExpr("CAST(value AS STRING) as value") \
                   .selectExpr(f"from_json(value, '{schema_str}') as json") \
                   .select("json.*") \
                   .dropDuplicates(["frame_id", "video_path"])

            pandas_df = df.toPandas()
            logger.debug(f"Batch {epoch_id}: Processed {len(pandas_df)} unique frames")

            if pandas_df.empty:
                logger.info(f"Batch {epoch_id}: Empty Pandas DataFrame, skipping processing")
                return

            # Batch decode and predict
            batch_size = 32
            frames = []
            valid_indices = []
            for i, row in pandas_df.iterrows():
                frame = self.process_frame(row['frame_data'])
                if frame is not None:
                    frames.append(frame)
                    valid_indices.append(i)

            if not frames:
                logger.warning(f"Batch {epoch_id}: No valid frames decoded")
                return

            scores = self.detector.predict_batch(frames)

            # Map scores efficiently
            results = []
            score_idx = 0
            for i, row in pandas_df.iterrows():
                result = {
                    "frame_id": row['frame_id'],
                    "timestamp": row['timestamp'],
                    "deepfake_score": float(scores[score_idx]) if i in valid_indices else 0.0,
                    "video_path": row['video_path']
                }
                results.append({"value": json.dumps(result)})
                if i in valid_indices:
                    score_idx += 1

            result_df = self.spark.createDataFrame(results, schema=StructType([StructField("value", StringType(), False)]))
            result_df.write \
                .format("kafka") \
                .option("kafka.bootstrap.servers", "localhost:29092") \
                .option("topic", "analysis-results") \
                .save()

            logger.info(f"Batch {epoch_id}: Processed {len(frames)} frames, wrote {len(results)} results")
        except Exception as e:
            logger.error(f"Batch {epoch_id}: Error processing batch: {str(e)}")
            raise

    def start_streaming(self, kafka_topic="video-frames", output_topic="analysis-results"):
        """Start Spark streaming to process video frames from Kafka."""
        try:
            df = self.spark.readStream \
                .format("kafka") \
                .option("kafka.bootstrap.servers", "localhost:29092") \
                .option("subscribe", kafka_topic) \
                .option("startingOffsets", "latest") \
                .option("failOnDataLoss", "false") \
                .option("maxOffsetsPerTrigger", "622") \
                .load()

            query = df.writeStream \
                .foreachBatch(self.process_batch) \
                .outputMode("append") \
                .trigger(processingTime="2 seconds") \
                .start()

            logger.info(f"Started Spark streaming from topic {kafka_topic} to {output_topic}")
            return query
        except Exception as e:
            logger.error(f"Error starting Spark streaming: {str(e)}")
            raise

    def stop(self):
        """Stop the Spark session and clean up."""
        try:
            self.spark.stop()
            logger.info("Spark session stopped")
        except Exception as e:
            logger.error(f"Error stopping Spark session: {str(e)}")

if __name__ == "__main__":
    processor = SparkVideoProcessor()
    query = processor.start_streaming()
    query.awaitTermination()