import streamlit as st
import cv2
from video_processor import VideoProducer
from spark_video_processor import SparkVideoProcessor, SparkTranscriptionProcessor
import threading
import time
import requests
import whisper
from kafka.admin import KafkaAdminClient, NewTopic
from kafka.errors import TopicAlreadyExistsError
import json
import pandas as pd
import os
import numpy as np
import torch
from optimized_deepfake_detector import OptimizedDeepfakeDetector
import plotly.graph_objects as go
from datetime import datetime, timedelta
import logging
import psutil
import altair as alt
from prometheus_client import Counter, Gauge, Histogram, start_http_server, REGISTRY, CollectorRegistry
from typing import Dict, Any, Optional, Tuple, List
from monitoring import MetricsCollector
import asyncio
import aiohttp
from dataclasses import dataclass
from functools import lru_cache
import re
from dotenv import load_dotenv
import google.generativeai as genai
from transformers import pipeline
import networkx as nx
import matplotlib.pyplot as plt
from pyvis.network import Network
from pathlib import Path
import sys
from concurrent.futures import ThreadPoolExecutor
import folium
import geopandas as gpd
from geopy.geocoders import Nominatim
from pymongo import MongoClient
from streamlit_folium import folium_static
from deep_translator import GoogleTranslator
from langdetect import detect

load_dotenv()
translator = GoogleTranslator(source='auto', target='en')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('fact_checker.log')
    ]
)
logger = logging.getLogger('veristream')

CACHE_TTL = 3600
MAX_RETRIES = 3
TIMEOUT = 30
BATCH_SIZE = 5

mongodb_uri = os.getenv("MONGODB_URI")
db_name = os.getenv("MONGODB_DB_NAME")
collection_name = os.getenv("MONGODB_COLLECTION_NAME")

LANGUAGE_MAPPING = {
    "Afrikaans": "af",
    "Albanian": "sq",
    "Amharic": "am",
    "Arabic": "ar",
    "Armenian": "hy",
    "Assamese": "as",
    "Aymara": "ay",
    "Azerbaijani": "az",
    "Bambara": "bm",
    "Basque": "eu",
    "Belarusian": "be",
    "Bengali": "bn",
    "Bhojpuri": "bho",
    "Bosnian": "bs",
    "Bulgarian": "bg",
    "Catalan": "ca",
    "Cebuano": "ceb",
    "Chichewa": "ny",
    "Chinese (Simplified)": "zh-CN",
    "Chinese (Traditional)": "zh-TW",
    "Corsican": "co",
    "Croatian": "hr",
    "Czech": "cs",
    "Danish": "da",
    "Dhivehi": "dv",
    "Dogri": "doi",
    "Dutch": "nl",
    "English": "en",
    "Esperanto": "eo",
    "Estonian": "et",
    "Ewe": "ee",
    "Filipino": "tl",
    "Finnish": "fi",
    "French": "fr",
    "Frisian": "fy",
    "Galician": "gl",
    "Georgian": "ka",
    "German": "de",
    "Greek": "el",
    "Guarani": "gn",
    "Gujarati": "gu",
    "Haitian Creole": "ht",
    "Hausa": "ha",
    "Hawaiian": "haw",
    "Hebrew": "iw",
    "Hindi": "hi",
    "Hmong": "hmn",
    "Hungarian": "hu",
    "Icelandic": "is",
    "Igbo": "ig",
    "Ilocano": "ilo",
    "Indonesian": "id",
    "Irish": "ga",
    "Italian": "it",
    "Japanese": "ja",
    "Javanese": "jw",
    "Kannada": "kn",
    "Kazakh": "kk",
    "Khmer": "km",
    "Kinyarwanda": "rw",
    "Konkani": "gom",
    "Korean": "ko",
    "Krio": "kri",
    "Kurdish (Kurmanji)": "ku",
    "Kurdish (Sorani)": "ckb",
    "Kyrgyz": "ky",
    "Lao": "lo",
    "Latin": "la",
    "Latvian": "lv",
    "Lingala": "ln",
    "Lithuanian": "lt",
    "Luganda": "lg",
    "Luxembourgish": "lb",
    "Macedonian": "mk",
    "Maithili": "mai",
    "Malagasy": "mg",
    "Malay": "ms",
    "Malayalam": "ml",
    "Maltese": "mt",
    "Maori": "mi",
    "Marathi": "mr",
    "Meiteilon (Manipuri)": "mni-Mtei",
    "Mizo": "lus",
    "Mongolian": "mn",
    "Myanmar": "my",
    "Nepali": "ne",
    "Norwegian": "no",
    "Odia (Oriya)": "or",
    "Oromo": "om",
    "Pashto": "ps",
    "Persian": "fa",
    "Polish": "pl",
    "Portuguese": "pt",
    "Punjabi": "pa",
    "Quechua": "qu",
    "Romanian": "ro",
    "Russian": "ru",
    "Samoan": "sm",
    "Sanskrit": "sa",
    "Scots Gaelic": "gd",
    "Sepedi": "nso",
    "Serbian": "sr",
    "Sesotho": "st",
    "Shona": "sn",
    "Sindhi": "sd",
    "Sinhala": "si",
    "Slovak": "sk",
    "Slovenian": "sl",
    "Somali": "so",
    "Spanish": "es",
    "Sundanese": "su",
    "Swahili": "sw",
    "Swedish": "sv",
    "Tajik": "tg",
    "Tamil": "ta",
    "Tatar": "tt",
    "Telugu": "te",
    "Thai": "th",
    "Tigrinya": "ti",
    "Tsonga": "ts",
    "Turkish": "tr",
    "Turkmen": "tk",
    "Twi": "ak",
    "Ukrainian": "uk",
    "Urdu": "ur",
    "Uyghur": "ug",
    "Uzbek": "uz",
    "Vietnamese": "vi",
    "Welsh": "cy",
    "Xhosa": "xh",
    "Yiddish": "yi",
    "Yoruba": "yo",
    "Zulu": "zu",
}

SENTIMENT_LABEL_MAPPING = {
    "LABEL_0": "Negative",
    "LABEL_1": "Neutral",
    "LABEL_2": "Positive"
}

if not mongodb_uri or not db_name or not collection_name:
    raise ValueError("Missing required environment variables. Please check your .env file.")


metrics = MetricsCollector(port=8000)

@dataclass
class AnalysisResult:
    """Data class to store analysis results"""
    text: str
    sentiment: Dict
    fact_checks: List[Dict]
    emotional_triggers: List[str]
    stereotypes: List[str]
    manipulation_score: float
    processing_time: float
    entities: List[Dict]
    locations: List[Dict]
    knowledge_graph: Optional[Dict] = None
    generative_analysis: Optional[Dict] = None

class KnowledgeGraphManager:
    def __init__(self):
        self.graph = nx.DiGraph()

    def get_verification_status(self, fact_checks: List[Dict]) -> tuple[str, str]:
        """Determine verification status from fact checks"""
        if not fact_checks:
            return "Not Confirmed", "No fact checks available"
        
        latest_check = fact_checks[0]
        if 'claimReview' in latest_check and latest_check['claimReview']:
            review = latest_check['claimReview'][0]
            rating = review.get('textualRating', '').lower()
            
            if any(word in rating for word in ['true', 'correct', 'accurate']):
                return "Verified True", review.get('textualRating', '')
            elif any(word in rating for word in ['false', 'incorrect', 'inaccurate']):
                return "Verified False", review.get('textualRating', '')
            else:
                return "Partially Verified", review.get('textualRating', '')
        
        return "Not Confirmed", "No clear verification status"

    def add_fact(self, text: str, entities: List[Dict], fact_checks: List[Dict], sentiment: Dict):
        """Add a fact and its related information to the knowledge graph"""
        fact_id = f"fact_{hash(text)}"
        self.graph.add_node(fact_id, 
                          type='fact',
                          text=text,
                          sentiment=sentiment.get('label', 'NEUTRAL'))
        
        verification_status, verification_details = self.get_verification_status(fact_checks)
        
        verification_id = f"verification_{fact_id}"
        self.graph.add_node(verification_id,
                         type='verification',
                         status=verification_status,
                         details=verification_details)
        
        self.graph.add_edge(fact_id, verification_id, relation='verified_as')
        
        for entity in entities:
            entity_id = f"entity_{hash(entity['text'])}"
            self.graph.add_node(entity_id,
                             type='entity',
                             text=entity['text'],
                             entity_type=entity['type'])
            
            self.graph.add_edge(entity_id, fact_id, relation='mentioned_in')
            
            self.graph.add_edge(entity_id, verification_id, 
                              relation=f"verified_{verification_status.lower().replace(' ', '_')}")
        
        for i, check in enumerate(fact_checks):
            if 'claimReview' in check and check['claimReview']:
                review = check['claimReview'][0]
                check_id = f"check_{fact_id}_{i}"
                self.graph.add_node(check_id,
                                 type='fact_check',
                                 source=review.get('publisher', {}).get('name', 'Unknown'),
                                 rating=review.get('textualRating', 'Unknown'),
                                 url=review.get('url', ''))
                self.graph.add_edge(verification_id, check_id, relation='supported_by')

    def get_graph_data(self) -> Dict:
        """Return graph data in a format suitable for visualization"""
        return {
            'nodes': list(self.graph.nodes(data=True)),
            'edges': list(self.graph.edges(data=True))
        }

    def visualize_graph(self, output_file: str = 'knowledge_graph.html'):
        """Create interactive and static visualizations of the knowledge graph"""
        colors = {
            'fact': '#ff7f7f',
            'entity': '#7f7fff',
            'verification': {
                'Verified True': '#00ff00',
                'Verified False': '#ff0000',
                'Partially Verified': '#ffff00',
                'Not Confirmed': '#808080'
            },
            'fact_check': '#7fff7f'
        }

        try:
            net = Network(height='750px', width='100%', bgcolor='#ffffff', font_color='#000000')
            net.barnes_hut()

            for node, data in self.graph.nodes(data=True):
                title = f"Type: {data['type']}<br>"
            
                if data['type'] == 'fact':
                    title += f"Text: {data['text']}<br>Sentiment: {data['sentiment']}"
                    node_color = colors['fact']
                elif data['type'] == 'entity':
                    title += f"Text: {data['text']}<br>Entity Type: {data['entity_type']}"
                    node_color = colors['entity']
                elif data['type'] == 'verification':
                    title += f"Status: {data['status']}<br>Details: {data['details']}"
                    node_color = colors['verification'].get(data['status'], '#808080')
                elif data['type'] == 'fact_check':
                    title += f"Source: {data['source']}<br>Rating: {data['rating']}"
                    if 'url' in data:
                        title += f"<br>URL: {data['url']}"
                    node_color = colors['fact_check']
            
                net.add_node(str(node), 
                            title=title,
                            color=node_color,
                            size=30 if data['type'] in ['verification', 'fact'] else 20)

            for edge in self.graph.edges(data=True):
                net.add_edge(str(edge[0]), 
                            str(edge[1]), 
                            title=edge[2].get('relation', ''),
                            physics=True)

            physics_settings = {
                "physics": {
                    "barnesHut": {
                        "gravitationalConstant": -2000,
                        "centralGravity": 0.3,
                        "springLength": 200,
                        "springConstant": 0.04,
                        "damping": 0.09,
                        "avoidOverlap": 0.1
                    },
                    "minVelocity": 0.75
                }
            }
            net.set_options(json.dumps(physics_settings))

            try:
                net.save_graph(output_file)
            except Exception as e:
                logger.error(f"Failed to save interactive visualization: {e}")

        except Exception as e:
            logger.error(f"Failed to create interactive visualization: {e}")

        try:
            plt.figure(figsize=(15, 10))
            pos = nx.spring_layout(self.graph, k=1, iterations=50)
        
            node_types = ['fact', 'entity', 'verification', 'fact_check']
            for node_type in node_types:
                if node_type == 'verification':
                    for status, color in colors['verification'].items():
                        node_list = [node for node, attr in self.graph.nodes(data=True)
                                   if attr['type'] == 'verification' and attr['status'] == status]
                        if node_list:
                            nx.draw_networkx_nodes(self.graph, pos,
                                                nodelist=node_list,
                                                node_color=color,
                                                node_size=2000,
                                                alpha=0.7)
                else:
                    node_list = [node for node, attr in self.graph.nodes(data=True)
                                if attr['type'] == node_type]
                    if node_list:
                        nx.draw_networkx_nodes(self.graph, pos,
                                            nodelist=node_list,
                                            node_color=colors[node_type],
                                            node_size=1500 if node_type == 'fact' else 1000,
                                            alpha=0.7)
        
            nx.draw_networkx_edges(self.graph, pos, 
                                 edge_color='gray', 
                                 arrows=True, 
                                 arrowsize=20,
                                 alpha=0.5)
        
            labels = {}
            for node, data in self.graph.nodes(data=True):
                if data['type'] == 'fact':
                    labels[node] = data['text'][:20] + '...'
                elif data['type'] == 'entity':
                    labels[node] = data['text']
                elif data['type'] == 'verification':
                    labels[node] = data['status']
                elif data['type'] == 'fact_check':
                    labels[node] = data['source'][:20]
        
            nx.draw_networkx_labels(self.graph, pos, labels, font_size=8)
        
            legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                                        markerfacecolor=color, label=f"Node: {label}", markersize=10)
                             for label, color in colors.items() if isinstance(color, str)]
        
            for status, color in colors['verification'].items():
                legend_elements.append(plt.Line2D([0], [0], marker='o', color='w',
                                                markerfacecolor=color, label=f"Status: {status}",
                                                markersize=10))
        
            plt.legend(handles=legend_elements, 
                      loc='center left', 
                      bbox_to_anchor=(1, 0.5),
                      fontsize=8)
        
            plt.title("Knowledge Graph Visualization")
            plt.axis('off')
            plt.tight_layout()
        
            static_output = 'knowledge_graph_static.png'
            plt.savefig(static_output, bbox_inches='tight', dpi=300)
            plt.close()

            html_content = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>Knowledge Graph Visualization</title>
               <style>
                    body {{
                        font-family: Arial, sans-serif;
                        margin: 20px;
                        background-color: #f5f5f5;
                    }}
                    .container {{
                        max-width: 1200px;
                        margin: 0 auto;
                        background-color: white;
                        padding: 20px;
                        border-radius: 8px;
                        box-shadow: 0 0 10px rgba(0,0,0,0.1);
                    }}
                    .visualization {{
                        text-align: center;
                        margin-bottom: 30px;
                    }}
                    img {{
                        max-width: 100%;
                        height: auto;
                        border-radius: 4px;
                    }}
                    .stats {{
                        display: grid;
                        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                        gap: 20px;
                        margin-bottom: 30px;
                    }}
                    .stat-card {{
                        background-color: #f8f9fa;
                        padding: 15px;
                        border-radius: 4px;
                        text-align: center;
                    }}
                    .nodes-list {{
                        max-height: 400px;
                        overflow-y: auto;
                        border: 1px solid #ddd;
                        padding: 15px;
                        border-radius: 4px;
                    }}
                </style>
            </head>
            <body>
                <div class="container">
                    <h1>Knowledge Graph Visualization</h1>
                
                    <div class="visualization">
                        <h2>Interactive Visualization</h2>
                        <iframe src="{output_file}" width="100%" height="800px" frameborder="0"></iframe>
                    </div>

                    <div class="visualization">
                        <h2>Static Visualization</h2>
                        <img src="{static_output}" alt="Static Knowledge Graph">
                    </div>
                
                    <div class="stats">
                        <div class="stat-card">
                            <h3>Total Nodes</h3>
                            <p>{len(self.graph.nodes)}</p>
                        </div>
                        <div class="stat-card">
                            <h3>Total Edges</h3>
                            <p>{len(self.graph.edges)}</p>
                        </div>
                        <div class="stat-card">
                            <h3>Node Types</h3>
                            <p>{len(set(data['type'] for _, data in self.graph.nodes(data=True)))}</p>
                        </div>
                    </div>
                
                    <div class="nodes-list">
                        <h2>Node Details</h2>
                        <ul>
            """
        
            for node, data in self.graph.nodes(data=True):
                html_content += f"<li><strong>{node}</strong>: {str(data)}</li>"
        
            html_content += """
                    </ul>
                </div>
            </div>
        </body>
        </html>
        """
        
            combined_output = 'knowledge_graph_combined.html'
            with open(combined_output, 'w', encoding='utf-8') as f:
                f.write(html_content)

        except Exception as e:
            logger.error(f"Failed to create static visualization: {e}")
            raise

        return {
            'interactive': output_file,
            'static': static_output,
            'combined': combined_output
        }
    
class MongoDBManager:
    def __init__(self, uri: str, db_name: str, collection_name: str):
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
    
mongo_manager = MongoDBManager(mongodb_uri, db_name, collection_name)


class OptimizedAnalyzer:
    def __init__(self, use_gpu: bool = False):
        """Initialize the analyzer with all required components"""
        self.fact_check_api_key = os.getenv('FACT_CHECK_API_KEY')
        self.logger = logging.getLogger('analyzer')
        self.device = 'cuda' if use_gpu and torch.cuda.is_available() else 'cpu'

        self.fact_check_api_key = os.getenv('FACT_CHECK_API_KEY')
        self.google_api_key = os.getenv('GOOGLE_API_KEY')
        self.session = None
        self.cache_file = Path('analysis_cache.json')
        self.knowledge_graph = KnowledgeGraphManager()
        self._load_cache()
        self._initialize_ml_pipelines()
        self._setup_detection_patterns()
        self._setup_generative_ai()

    def _initialize_ml_pipelines(self):
        """Initialize all ML pipelines with optimized batch processing"""
        try:
            self.logger.info("Initializing ML pipelines...")
            
            self.sentiment_pipeline = pipeline(
                "sentiment-analysis",
                model="cardiffnlp/twitter-roberta-base-sentiment",
                device=self.device,
                batch_size=BATCH_SIZE
            )
            
            self.ner_pipeline = pipeline(
                task="ner",
                model="dslim/bert-base-NER",
                aggregation_strategy="simple",
                device=self.device,
                batch_size=BATCH_SIZE
            )
            
            self.zero_shot_classifier = pipeline(
                task="zero-shot-classification",
                model="facebook/bart-large-mnli",
                device=self.device,
                batch_size=BATCH_SIZE
            )
            
            self.logger.info("ML pipelines initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize ML pipelines: {str(e)}")
            raise

    def _setup_detection_patterns(self):
        """Setup patterns for emotional and stereotype detection"""
        self.EMOTIONAL_TRIGGERS = [
            r"breaking news", r"fear of", r"unprecedented", r"urgent",
            r"shocking", r"critical", r"emergency", r"life-changing",
            r"act now", r"time is running out", r"don't miss out",
            r"last chance", r"limited time", r"before it's too late",
            r"alert", r"warning", r"danger", r"crisis", r"panic",
            r"disaster", r"catastrophe", r"chaos", r"threat", r"terror",
            r"amazing", r"incredible", r"unbelievable", r"mind-blowing",
            r"jaw-dropping", r"once in a lifetime", r"exclusive", r"secret",
            r"revealed", r"exposed", r"hidden truth", r"you won't believe",
            r"shocking truth", r"must see", r"must watch", r"viral",
            r"trending", r"explosive", r"sensational", r"scandal",
            r"you should", r"you must", r"you need to", r"it's your duty",
            r"responsibility", r"obligation", r"you owe it to", r"don't let down",
            r"disappoint", r"failure", r"let everyone down", r"guilt",
            r"shame", r"ashamed", r"regret", r"missed opportunity",
            r"outrageous", r"disgusting", r"appalling", r"unacceptable",
            r"infuriating", r"enraging", r"maddening", r"furious",
            r"angry", r"rage", r"fury", r"indignation", r"resentment",
            r"betrayal", r"treachery", r"hypocrisy", r"corruption",
            r"heartbreaking", r"tragic", r"devastating", r"tear-jerking",
            r"sob story", r"pitiful", r"miserable", r"depressing",
            r"hopeless", r"helpless", r"despair", r"grief", r"sorrow",
            r"mourn", r"loss", r"pain", r"suffering", r"anguish",
            r"don't miss", r"exclusive offer", r"limited edition",
            r"only a few left", r"while supplies last", r"free gift",
            r"no risk", r"guaranteed", r"proven", r"scientifically proven",
            r"miracle", r"instant results", r"overnight success",
            r"secret method", r"hidden trick", r"loophole", r"hack",
            r"cheat code", r"get rich quick", r"lose weight fast",
        ]
        
        self.STEREOTYPE_PATTERNS = [
            r"women can't", r"men are always", r"women should", r"men should",
            r"women belong in", r"men belong in", r"women are too", r"men are too",
            r"women are naturally", r"men are naturally", r"women are better at",
            r"men are better at", r"women are worse at", r"men are worse at",
            r"women are emotional", r"men are emotional", r"women are weak",
            r"men are strong", r"women are submissive", r"men are dominant",
            r"all \w+s are", r"\w+ people always", r"typical \w+ behavior",
            r"\w+ people can't", r"\w+ people are lazy", r"\w+ people are greedy",
            r"\w+ people are violent", r"\w+ people are criminals",
            r"\w+ people are uneducated", r"\w+ people are poor",
            r"\w+ people are rich", r"\w+ people are cheap",
            r"\w+ people are aggressive", r"\w+ people are submissive",
            r"\w+ people are exotic", r"\w+ people are primitive",
            r"young people are", r"old people are", r"millennials are",
            r"boomers are", r"gen z are", r"teenagers are", r"kids these days",
            r"back in my day", r"young people don't", r"old people can't",
            r"young people are lazy", r"old people are slow",
            r"young people are entitled", r"old people are out of touch",
            r"\w+ people are fanatics", r"\w+ people are intolerant",
            r"\w+ people are extremists", r"\w+ people are terrorists",
            r"\w+ people are backward", r"\w+ people are superstitious",
            r"\w+ people are closed-minded", r"\w+ people are oppressive",
            r"\w+ people are rude", r"\w+ people are arrogant",
            r"\w+ people are lazy", r"\w+ people are hardworking",
            r"\w+ people are dishonest", r"\w+ people are corrupt",
            r"\w+ people are violent", r"\w+ people are peaceful",
            r"\w+ people are greedy", r"\w+ people are generous",
            r"all lawyers are", r"all doctors are", r"all teachers are",
            r"all politicians are", r"all cops are", r"all artists are",
            r"all engineers are", r"all scientists are", r"all bankers are",
            r"all journalists are", r"all athletes are", r"all actors are",
            r"poor people are", r"rich people are", r"homeless people are",
            r"disabled people are", r"immigrants are", r"refugees are",
            r"foreigners are", r"locals are", r"city people are",
            r"country people are", r"educated people are",
            r"uneducated people are", r"liberals are", r"conservatives are",
        ]

    def _setup_generative_ai(self):
        """Setup Google's Generative AI if API key is available"""
        if self.google_api_key:
            try:
                genai.configure(api_key=self.google_api_key)
                self.generative_model = genai.GenerativeModel('gemini-pro')
            except Exception as e:
                self.logger.error(f"Generative AI setup failed: {e}")
                self.generative_model = None
        else:
            self.generative_model = None

    async def _init_session(self):
        """Initialize aiohttp session"""
        if self.session is None:
            self.session = aiohttp.ClientSession()

    async def _close_session(self):
        """Close aiohttp session"""
        if self.session:
            await self.session.close()
            self.session = None

    def _load_cache(self):
        """Load cache from file with error handling"""
        try:
            if self.cache_file.exists():
                with open(self.cache_file, 'r') as f:
                    self.cache = json.load(f)
            else:
                self.cache = {}
        except Exception as e:
            self.logger.error(f"Error loading cache: {str(e)}")
            self.cache = {}

    @lru_cache(maxsize=1000)
    def detect_patterns(self, text: str) -> tuple[List[str], List[str]]:
        """Detect emotional triggers and stereotypes with caching"""
        emotional_triggers = [
            trigger for trigger in self.EMOTIONAL_TRIGGERS
            if re.search(trigger, text, re.IGNORECASE)
        ]
        
        stereotypes = [
            pattern for pattern in self.STEREOTYPE_PATTERNS
            if re.search(pattern, text, re.IGNORECASE)
        ]
        
        return emotional_triggers, stereotypes

    def fact_check(self, text: str) -> List[Dict]:
        """Perform async fact checking"""
        if not self.fact_check_api_key:
            logger.warning("Fact check API key is missing")
            print("Fact check API key is missing")
            return []

        cache_key = f"fact_check_{text}"
        if cache_key in self.cache:
            cache_entry = self.cache[cache_key]
            if time() - cache_entry['timestamp'] < CACHE_TTL:
                return cache_entry['data']
            
        base_url = "https://factchecktools.googleapis.com/v1alpha1/claims:search"
        params = {
            'query': text,
            'key': self.fact_check_api_key
        }

        for attempt in range(MAX_RETRIES):
            try:
                response = requests.get(base_url, params=params, timeout=TIMEOUT)
                response.raise_for_status()  
                data = response.json()
                self.cache[cache_key] = {
                    'data': data.get('claims', []),
                    'timestamp': time.time()
                }
                return data.get('claims', [])
            except requests.exceptions.RequestException as e:
                logger.error(f"Fact check attempt {attempt + 1} failed: {str(e)}")
                time.sleep(1 * (attempt + 1))
        
        return []

    def analyze_text(self, text: str) -> AnalysisResult:
        """Perform comprehensive text analysis"""
        start_time = time.time()

        try:
            fact_checks = self.fact_check(text)
            sentiment = self.sentiment_pipeline(text)
            ner_result = self.ner_pipeline(text)
            classification = self.zero_shot_classifier(text, ["true claim", "false claim"])
            emotional_triggers, stereotypes = self.detect_patterns(text)

            entities = [{
                'text': entity['word'],
                'type': entity['entity_group']
            } for entity in ner_result]

            locations = [entity for entity in entities if entity['type'] == 'LOC']

            self.knowledge_graph.add_fact(text, entities, fact_checks, sentiment[0] if sentiment else {})

            manipulation_score = self._compute_manipulation_score(
                sentiment,
                fact_checks,
                emotional_triggers,
                stereotypes
            )

            generative_analysis = None
            if self.generative_model:
                try:
                    response = self.generative_model.generate_content(
                        f"Analyze the sentiment and credibility of: {text}"
                    )
                    generative_analysis = {"analysis": response.text}
                except Exception as e:
                    self.logger.error(f"Generative analysis failed: {e}")

            return AnalysisResult(
                text=text,
                sentiment=sentiment[0] if sentiment else {},
                fact_checks=fact_checks,
                emotional_triggers=emotional_triggers,
                stereotypes=stereotypes,
                manipulation_score=manipulation_score,
                processing_time=time.time() - start_time,
                entities=entities,
                locations=locations,
                knowledge_graph=self.knowledge_graph.get_graph_data(),
                generative_analysis=generative_analysis
            )

        except Exception as e:
            self.logger.error(f"Analysis failed: {str(e)}")
            raise

    def _compute_manipulation_score(self, 
                                 sentiment: List[Dict],
                                 fact_checks: List,
                                 emotional_triggers: List,
                                 stereotypes: List) -> float:
        """Compute manipulation risk score with weighted factors"""
        try:
            sentiment_weight = 0.3
            fact_weight = 0.3
            trigger_weight = 0.2
            stereotype_weight = 0.2
            
            sentiment_score = 1.0 if sentiment and sentiment[0]['label'] == 'NEGATIVE' else 0.0
            fact_score = 1.0 if not fact_checks else 0.0
            trigger_score = min(len(emotional_triggers) * 0.2, 1.0)
            stereotype_score = min(len(stereotypes) * 0.2, 1.0)
            
            final_score = (
                sentiment_score * sentiment_weight +
                fact_score * fact_weight +
                trigger_score * trigger_weight +
                stereotype_score * stereotype_weight
            )
            
            return round(final_score, 2)
        except Exception as e:
            self.logger.error(f"Manipulation score computation failed: {e}")
            return 0.5

@st.cache_resource
def load_models() -> Tuple[Any, OptimizedDeepfakeDetector]:
    try:
        whisper_model = whisper.load_model("base")
        detector = OptimizedDeepfakeDetector()
        return whisper_model, detector
    except Exception as e:
        logger.error(f"Error loading models: {e}")
        metrics.system_healthy.set(0)
        raise

class VideoAnalyzer:
    def __init__(self):
        self.whisper_model, self.detector = load_models()
        self.producer = VideoProducer()
        self.spark_processor = SparkVideoProcessor()
        self.metrics = metrics

    def process_frame_batch(self, frames: List[np.ndarray]) -> Tuple[float, float, int]:
        try:
            start_time = time.time()
            avg_score, max_score = self.detector.predict_batch(frames)
            processing_time = time.time() - start_time
            
            self.metrics.record_metric('processing_time', float(processing_time))
            self.metrics.record_metric('frames_processed', float(len(frames)))
            
            return float(avg_score), float(max_score), len(frames)
        except Exception as e:
            logger.error(f"Error processing frame batch: {e}")
            self.metrics.system_healthy.set(0)
            return 0.0, 0.0, 0

    def analyze_video(self, video_path: str, progress_bar) -> Tuple[str, float, Dict]:
        try:
            cap = cv2.VideoCapture(video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            
            frames_data = {
                'scores': [],
                'max_scores': [],
                'timestamps': [],
                'faces_detected': []
            }
            
            batch_size = 32
            frames_batch = []
            frame_count = 0
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                    
                frames_batch.append(frame)
                frame_count += 1
                
                if len(frames_batch) >= batch_size:
                    avg_score, max_score, faces = self.process_frame_batch(frames_batch)
                    frames_data['scores'].append(float(avg_score))
                    frames_data['max_scores'].append(float(max_score))
                    frames_data['timestamps'].append(float(frame_count) / fps)
                    frames_data['faces_detected'].append(int(faces))
                    frames_batch = []
                    
                progress_bar.progress(min(frame_count / total_frames, 0.5))
                
            cap.release()
            
            if frames_batch:
                avg_score, max_score, faces = self.process_frame_batch(frames_batch)
                frames_data['scores'].append(float(avg_score))
                frames_data['max_scores'].append(float(max_score))
                frames_data['timestamps'].append(float(frame_count) / fps)
                frames_data['faces_detected'].append(int(faces))
            
            progress_bar.progress(0.7)
            streaming_query = self.spark_processor.start_streaming(self.detector)
            transcription = self.whisper_model.transcribe(video_path)
            self.producer.send_video(video_path)
            
            final_score = float(np.mean(frames_data['max_scores']))
            return transcription["text"], final_score, frames_data
            
        except Exception as e:
            logger.error(f"Error analyzing video: {e}")
            self.metrics.system_healthy.set(0)
            raise

def create_gis_map():
    """Create a Folium map with markers for the detected locations."""
    geolocator = Nominatim(user_agent="geoapiExercises")
    map_center = [20.5937, 78.9629]  
    m = folium.Map(location=map_center, zoom_start=5)

    locations = mongo_manager.get_all_locations()
    
    for location in locations:
        try:
            geo_location = geolocator.geocode(location)
            if geo_location:
                folium.Marker(
                    location=[geo_location.latitude, geo_location.longitude],
                    popup=location,
                    icon=folium.Icon(color='red', icon='info-sign')
                ).add_to(m)
        except Exception as e:
            logger.error(f"Error geocoding location {location}: {e}")
    
    return m


def create_monitoring_dashboard():
    st.subheader("System Monitoring")
    
    col1, col2, col3 = st.columns(3)
    
    try:
        system_metrics = metrics.get_system_metrics()
        
        with col1:
            cpu_usage = system_metrics['cpu_usage']
            st.metric("CPU Usage", f"{cpu_usage:.1f}%",
                     delta=f"{cpu_usage - 50:.1f}%" if cpu_usage > 50 else None,
                     delta_color="inverse")
            
        with col2:
            memory_usage = system_metrics['memory_usage']
            st.metric("Memory Usage", f"{memory_usage:.1f}%",
                     delta=f"{memory_usage - 70:.1f}%" if memory_usage > 70 else None,
                     delta_color="inverse")
            
        with col3:
            system_healthy = system_metrics['system_healthy']
            health_status = "Healthy" if system_healthy > 0.5 else "Unhealthy"
            health_delta = "OK" if system_healthy > 0.5 else "Check Logs"
            st.metric("System Health", health_status,
                     delta=health_delta,
                     delta_color="normal" if health_status == "Healthy" else "inverse")
        
    except Exception as e:
        logger.error(f"Error creating monitoring dashboard: {e}", exc_info=True)
        st.error("Unable to display monitoring dashboard")

def display_analysis_results(final_score: float, frames_data: Dict):
    col1, col2 = st.columns(2)
    
    with col1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=frames_data['timestamps'],
            y=frames_data['scores'],
            mode='lines',
            name='Average Score',
            line=dict(color='blue', width=2)
        ))
        fig.add_trace(go.Scatter(
            x=frames_data['timestamps'],
            y=frames_data['max_scores'],
            mode='lines',
            name='Max Score',
            line=dict(color='red', width=2)
        ))
        fig.update_layout(
            title='Deepfake Detection Confidence Over Time',
            xaxis_title='Time (seconds)',
            yaxis_title='Confidence Score',
            hovermode='x unified'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        hist_fig = go.Figure()
        hist_fig.add_trace(go.Histogram(
            x=frames_data['max_scores'],
            nbinsx=30,
            name='Score Distribution',
            marker_color='rgb(55, 83, 109)'
        ))
        hist_fig.update_layout(
            title='Detection Score Distribution',
            xaxis_title='Confidence Score',
            yaxis_title='Frequency',
            bargap=0.1
        )
        st.plotly_chart(hist_fig, use_container_width=True)
    
    st.subheader("Analysis Summary")
    summary_col1, summary_col2, summary_col3 = st.columns(3)
    
    with summary_col1:
        avg_score = float(np.mean(frames_data['max_scores']))
        st.metric(
            "Average Confidence",
            f"{avg_score:.2%}",
            delta=f"{avg_score - 0.5:.2%}",
            delta_color="inverse"
        )
    
    with summary_col2:
        max_score = float(max(frames_data['max_scores']))
        st.metric(
            "Peak Detection Score",
            f"{max_score:.2%}",
            delta=f"{max_score - avg_score:.2%}",
            delta_color="inverse"
        )
    
    with summary_col3:
        total_frames = len(frames_data['timestamps'])
        st.metric(
            "Total Frames Analyzed",
            f"{total_frames:,}",
            f"{total_frames/30:.1f} seconds"
        )
    
    if final_score > 0.7:
        st.error(f"🚨 High probability of deepfake detected (Confidence: {final_score:.2%})")
        st.markdown("""
            **Detection Details:**
            - Multiple frames show signs of manipulation
            - High confidence scores sustained over time
            - Consistent detection patterns across video segments
            
            **Recommended Actions:**
            - Conduct manual review
            - Check video metadata
            - Verify source authenticity
        """)
    elif final_score > 0.4:
        st.warning(f"⚠️ Potential manipulation detected (Confidence: {final_score:.2%})")
        st.markdown("""
            **Analysis Notes:**
            - Some suspicious frames detected
            - Moderate confidence in manipulation
            - Further investigation recommended
        """)
    else:
        st.success(f"✅ Video appears authentic (Confidence: {1-final_score:.2%})")
        st.markdown("""
            **Analysis Notes:**
            - No significant manipulation patterns detected
            - Low confidence scores across frames
            - Normal video characteristics observed
        """)
    
    with st.expander("Detailed Metrics"):
        metrics_df = pd.DataFrame({
            'Time (s)': frames_data['timestamps'],
            'Average Score': frames_data['scores'],
            'Max Score': frames_data['max_scores'],
            'Faces Detected': frames_data['faces_detected']
        })
        st.dataframe(
            metrics_df.style.background_gradient(subset=['Max Score'], cmap='RdYlGn_r'),
            use_container_width=True
        )
        
        csv = metrics_df.to_csv(index=False)
        st.download_button(
            label="Download Detailed Results",
            data=csv,
            file_name="deepfake_analysis_results.csv",
            mime="text/csv"
        )

def visualize_knowledge_graph_interactive(graph_data):
    """
    Visualize the knowledge graph interactively using pyvis.
    
    Args:
        graph_data (dict): The knowledge graph data containing nodes and edges.
    """
    net = Network(notebook=True, directed=True, height="750px", width="100%", bgcolor="#ffffff", font_color="#000000")
    
    for node_id, node_data in graph_data["nodes"]:
        label = node_data.get("text", node_id)
        title = f"Type: {node_data.get('type', 'N/A')}\n"
        if node_data["type"] == "fact":
            title += f"Text: {node_data.get('text', 'N/A')}\nSentiment: {node_data.get('sentiment', 'N/A')}"
        elif node_data["type"] == "verification":
            title += f"Status: {node_data.get('status', 'N/A')}\nDetails: {node_data.get('details', 'N/A')}"
        elif node_data["type"] == "entity":
            title += f"Text: {node_data.get('text', 'N/A')}\nEntity Type: {node_data.get('entity_type', 'N/A')}"
        net.add_node(node_id, label=label, title=title, color=get_node_color(node_data["type"]))
    
    for edge in graph_data["edges"]:
        source, target, edge_data = edge
        net.add_edge(source, target, title=edge_data.get("relation", ""), color="gray")
    
    net.set_options("""
    {
        "physics": {
            "barnesHut": {
                "gravitationalConstant": -2000,
                "centralGravity": 0.3,
                "springLength": 200,
                "springConstant": 0.04,
                "damping": 0.09,
                "avoidOverlap": 0.1
            },
            "minVelocity": 0.75
        }
    }
    """)
    
    net.save_graph("knowledge_graph.html")
    st.components.v1.html(open("knowledge_graph.html", "r").read(), height=800)

def get_node_color(node_type):
    """Return a color based on the node type."""
    colors = {
        "fact": "#ff7f7f",  
        "verification": "#7f7fff",  
        "entity": "#7fff7f"  
    }
    return colors.get(node_type, "#808080")  

def main():
    st.title("VERISTREAM")
    st.markdown("### Real-time Deepfake Detection & Transcription")
    
    create_monitoring_dashboard()
    
    with st.sidebar:
        st.title("System Status")
        if st.button("Check Kafka Topics"):
            try:
                admin_client = KafkaAdminClient(bootstrap_servers=['localhost:29092'])
                topics = admin_client.list_topics()
                st.json({"Available Topics": topics})
            except Exception as e:
                st.error(f"Error checking topics: {e}")
                logger.error(f"Kafka error: {e}")
    
    uploaded_file = st.file_uploader("Upload Video for Analysis", type=['mp4', 'avi', 'mov'])

    if uploaded_file:
        temp_path = f"temp_{uploaded_file.name}"
        try:
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            video_file = open(temp_path, "rb")
            st.video(video_file)
            
            progress_bar = st.progress(0)
            st.write("Analyzing video...")
            
            try:
                analyzer = VideoAnalyzer()
                transcription, final_score, frames_data = analyzer.analyze_video(temp_path, progress_bar)
                
                text_analyzer = OptimizedAnalyzer(use_gpu=True)
                analysis_result = text_analyzer.analyze_text(transcription)
                
                display_analysis_results(final_score, frames_data)
                
                with st.expander("Video Transcription"):
                    st.write(transcription)
                    target_language = st.selectbox("Translate to", ["English", "Assamese","Bengali", "Gujarati","Hindi", "Kannada","Malayalam","Marathi","Odia (Oriya)","Urdu"])
    
                    if target_language != "English":
                        target_code = LANGUAGE_MAPPING.get(target_language, "en")
                        translated_text = GoogleTranslator(source='auto', target=target_code).translate(transcription)
                        st.write(f"Translated to {target_language}:")
                        st.write(translated_text)
                
                with st.expander("Text Analysis Results"):
                    st.write("### Sentiment Analysis")
                    if analysis_result.sentiment:
                        sentiment_label = analysis_result.sentiment["label"]
                        st.write(f"**Sentiment:** {sentiment_label}")
                        st.write(f"**Confidence Score:** {analysis_result.sentiment['score']:.4f}")
                    else:
                        st.write("Sentiment analysis not available")
                    
                    st.write("### Fact Checks")
                    st.write(analysis_result.fact_checks)
                    
                    st.write("### Emotional Triggers")
                    st.write(analysis_result.emotional_triggers)
                    
                    st.write("### Stereotypes")
                    st.write(analysis_result.stereotypes)
                    
                    st.write("### Manipulation Score")
                    st.write(analysis_result.manipulation_score)
                    
                    st.write("### Entities")
                    st.write(analysis_result.entities)
                    
                    st.write("### Knowledge Graph")
                    visualize_knowledge_graph_interactive(analysis_result.knowledge_graph)
                    
                    st.write("### Generative Analysis")
                    st.write(analysis_result.generative_analysis)

                    st.subheader("Geospatial Visualization of Detected Locations")

                    gis_map = create_gis_map()
                    folium_static(gis_map)
                
                
                progress_bar.progress(1.0)
                
            except Exception as e:
                st.error(f"Error processing video: {str(e)}")
                logger.error(f"Processing error: {str(e)}", exc_info=True)
                if "NoSuchMethodError" in str(e):
                    st.warning("Spark version compatibility issue detected. Please check system configurations.")
                return
            
        except Exception as e:
            st.error(f"Error processing video: {str(e)}")
            logger.error(f"Processing error: {str(e)}")
            metrics.system_healthy.set(0)
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

if __name__ == "__main__":
    main()