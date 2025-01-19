from pymongo import MongoClient
from datetime import datetime
import logging
from typing import Dict
from config import MONGODB_URI, MONGODB_DB_NAME, MONGODB_COLLECTION_NAME

logger = logging.getLogger('veristream')
class MongoDBManager:
    def __init__(self, uri: str = MONGODB_URI, db_name: str = MONGODB_DB_NAME, collection_name: str = MONGODB_COLLECTION_NAME):
        self.client = MongoClient(uri)
        self.db = self.client[db_name]
        self.collection = self.db[collection_name]

    def insert_location(self, location: str, metadata: Dict = None):
        """Insert a location into the MongoDB collection."""
        document = {
            'location': location,
            'timestamp': datetime.now(),
            'metadata': metadata if metadata else {}
        }
        self.collection.insert_one(document)

    def get_all_locations(self):
        """Retrieve all locations from the MongoDB collection."""
        return list(self.collection.find({}, {'_id': 0}))