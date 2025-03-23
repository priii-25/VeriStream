from pymongo import MongoClient
from datetime import datetime
import logging
from typing import Dict
from config import MONGODB_URI, MONGODB_DB_NAME, MONGODB_COLLECTION_NAME
import os  

logger = logging.getLogger('veristream')

class MongoDBManager:
    def __init__(self, uri: str = MONGODB_URI, db_name: str = MONGODB_DB_NAME, collection_name: str = MONGODB_COLLECTION_NAME):
        self.demo_mode = os.getenv("DEMO_MODE", "False").lower() == "true"

        if self.demo_mode:
            self.in_memory_data = []  
            logger.warning("Running in DEMO MODE. No real MongoDB connection.")
        else:
            try:
                self.client = MongoClient(uri)
                self.db = self.client[db_name]
                self.collection = self.db[collection_name]
            except Exception as e:
                logger.error(f"Failed to connect to MongoDB: {e}")
                self.demo_mode = True
                self.in_memory_data = []
                logger.warning("Falling back to DEMO MODE due to MongoDB connection error.")

    def insert_location(self, location: str, metadata: Dict = None):
        """Insert a location."""
        if self.demo_mode:
            self.in_memory_data.append({'location': location, 'timestamp': datetime.now(), 'metadata': metadata})
        else:
            document = {
                'location': location,
                'timestamp': datetime.now(),
                'metadata': metadata if metadata else {}
            }
            try:
                self.collection.insert_one(document)
            except Exception as e:
                 logger.error(f"Failed to insert into MongoDB: {e}")


    def get_all_locations(self):
        """Retrieve all locations."""
        if self.demo_mode:
            return self.in_memory_data
        else:
            try:
                return list(self.collection.find({}, {'_id': 0}))
            except Exception as e:
                logger.error(f"Failed to retrieve from MongoDB: {e}")
                return [] 