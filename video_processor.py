import os
from kafka import KafkaProducer, KafkaConsumer
import cv2
from base64 import b64encode
import json
import numpy as np
from datetime import datetime

class VideoProducer:
    def __init__(self):
        self.producer = KafkaProducer(
            bootstrap_servers=['localhost:29092'],
            value_serializer=lambda v: json.dumps(v).encode('utf-8'),
            max_request_size=10485760  
        )

    def video_to_frames(self, video_path, batch_size=30):
        """Convert video to batches of frames."""
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
            
        cap = cv2.VideoCapture(video_path)
        frames = []
        frame_count = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, (640, 480))
            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            encoded_frame = b64encode(buffer).decode('utf-8')
            
            frames.append({
                'frame_id': frame_count,
                'timestamp': datetime.now().isoformat(),
                'frame_data': encoded_frame
            })
            
            frame_count += 1
            
            if len(frames) == batch_size:
                yield frames
                frames = []
        
        if frames:
            yield frames
        
        cap.release()

    def send_video(self, video_path, topic_name="video-frames"):
        """Send video frames to a Kafka topic."""
        print(f"Starting to process video: {video_path}")
        try:
            for batch in self.video_to_frames(video_path):
                try:
                    future = self.producer.send(topic_name, value=batch)
                    future.get(timeout=10)
                    print(f"Sent batch of {len(batch)} frames")
                except Exception as e:
                    print(f"Error sending batch: {e}")
        except Exception as e:
            print(f"Error processing video: {e}")
        finally:
            self.producer.close()
            print("Producer closed")

def test_consumer():
    """Simple consumer to verify Kafka messages."""
    consumer = KafkaConsumer(
        'video-frames',
        bootstrap_servers=['localhost:29092'],
        value_deserializer=lambda x: json.loads(x.decode('utf-8')),
        auto_offset_reset='earliest',
        group_id='test-group'
    )
    
    print("Consumer started - waiting for messages...")
    for message in consumer:
        print(f"Received batch of {len(message.value)} frames")
