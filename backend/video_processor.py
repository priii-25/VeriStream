# backend/video_processor.py
import cv2
import logging
import os
from kafka import KafkaProducer
import json
import base64
from typing import List, Dict

logger = logging.getLogger(__name__)

class VideoProducer:
    def __init__(self):
        try:
            self.producer = KafkaProducer(
                bootstrap_servers=['localhost:29092'],
                value_serializer=lambda x: json.dumps(x).encode('utf-8'),
                max_request_size=5242880,  # 5MB max message size
                retries=3,  # Retry sending on failure
                acks='all'  # Wait for all replicas to acknowledge
            )
            logger.info("Kafka Producer initialized")
        except Exception as e:
            logger.error(f"Failed to initialize Kafka Producer: {str(e)}")
            raise

    def process_video(self, video_path: str, topic: str = "video-frames") -> Dict[str, int]:
        """Process video and send frames to Kafka."""
        if not os.path.exists(video_path):
            logger.error(f"Video file not found: {video_path}")
            raise FileNotFoundError(f"Video file not found: {video_path}")

        cap = cv2.VideoCapture(video_path)
        frame_count = 0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if not cap.isOpened():
            logger.error(f"Failed to open video file: {video_path}")
            raise ValueError(f"Failed to open video file: {video_path}")

        logger.info(f"Starting video processing: {video_path}, total frames: {total_frames}")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1

            # Resize frame and encode as base64
            frame = cv2.resize(frame, (640, 480))  # Optimize size
            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            frame_base64 = base64.b64encode(buffer).decode('utf-8')

            # Frame data to send to Kafka
            frame_data = {
                'frame_id': frame_count,
                'video_path': video_path,
                'timestamp': cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0,
                'frame_data': frame_base64
            }

            try:
                self.producer.send(topic, value=frame_data)
                if frame_count % 100 == 0:  # Log progress periodically
                    logger.info(f"Sent {frame_count}/{total_frames} frames to Kafka")
            except Exception as e:
                logger.error(f"Failed to send frame {frame_count} to Kafka: {str(e)}")

        cap.release()
        self.producer.flush()  # Ensure all messages are sent
        logger.info(f"Finished processing video: {video_path}, total frames sent: {frame_count}")
        return {"total_frames": frame_count}

    def close(self):
        """Close the Kafka producer."""
        if self.producer:
            self.producer.close()
            logger.info("Kafka Producer closed")

if __name__ == "__main__":
    producer = VideoProducer()
    producer.process_video("sample_video.mp4")
    producer.close()