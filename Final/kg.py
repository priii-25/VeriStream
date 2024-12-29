import asyncio
import aiohttp
import logging
import sys
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from time import time
from functools import lru_cache
import json
from pathlib import Path
import torch
from concurrent.futures import ThreadPoolExecutor
import os
import re
from dotenv import load_dotenv
import google.generativeai as genai
from transformers import pipeline
import networkx as nx
import matplotlib.pyplot as plt
from pyvis.network import Network
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('fact_checker.log')
    ]
)
logger = logging.getLogger('fact_checker')

# Constants
CACHE_TTL = 3600
MAX_RETRIES = 3
TIMEOUT = 30
BATCH_SIZE = 5

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
    knowledge_graph: Optional[Dict] = None
    generative_analysis: Optional[Dict] = None

class KnowledgeGraphManager:
    def __init__(self):
        self.graph = nx.DiGraph()
        
    def add_fact(self, text: str, entities: List[Dict], fact_checks: List[Dict], sentiment: Dict):
        """Add a fact and its related information to the knowledge graph"""
        # Add fact node
        fact_id = f"fact_{hash(text)}"
        self.graph.add_node(fact_id, 
                          type='fact',
                          text=text,
                          sentiment=sentiment.get('label', 'NEUTRAL'))
        
        # Add entity nodes and connect to fact
        for entity in entities:
            entity_id = f"entity_{hash(entity['text'])}"
            self.graph.add_node(entity_id,
                             type='entity',
                             text=entity['text'],
                             entity_type=entity['type'])
            self.graph.add_edge(entity_id, fact_id, relation='mentioned_in')
        
        # Add fact check nodes
        for i, check in enumerate(fact_checks):
            check_id = f"check_{fact_id}_{i}"
            self.graph.add_node(check_id,
                             type='fact_check',
                             source=check.get('claimReview', [{}])[0].get('publisher', {}).get('name', 'Unknown'),
                             rating=check.get('claimReview', [{}])[0].get('textualRating', 'Unknown'))
            self.graph.add_edge(fact_id, check_id, relation='verified_by')
    
    def get_graph_data(self) -> Dict:
        """Return graph data in a format suitable for visualization"""
        return {
            'nodes': list(self.graph.nodes(data=True)),
            'edges': list(self.graph.edges(data=True))
        }
    
    def visualize_graph(self, output_file: str = 'knowledge_graph.html'):
        """Create an interactive visualization of the knowledge graph"""
        net = Network(height='750px', width='100%', bgcolor='#ffffff', font_color='#000000')
        
        # Add nodes with different colors based on type
        colors = {'fact': '#ff7f7f', 'entity': '#7f7fff', 'fact_check': '#7fff7f'}
        
        for node, data in self.graph.nodes(data=True):
            title = f"Type: {data['type']}<br>"
            if data['type'] == 'fact':
                title += f"Text: {data['text']}<br>Sentiment: {data['sentiment']}"
            elif data['type'] == 'entity':
                title += f"Text: {data['text']}<br>Entity Type: {data['entity_type']}"
            elif data['type'] == 'fact_check':
                title += f"Source: {data['source']}<br>Rating: {data['rating']}"
            
            net.add_node(str(node), 
                        title=title,
                        color=colors[data['type']])
        
        # Add edges
        for edge in self.graph.edges(data=True):
            net.add_edge(str(edge[0]), str(edge[1]))
        
        # Generate the visualization
        net.show(output_file)
        
        # Also create a static matplotlib visualization
        plt.figure(figsize=(12, 8))
        pos = nx.spring_layout(self.graph)
        
        # Draw nodes
        for node_type, color in colors.items():
            node_list = [node for node, attr in self.graph.nodes(data=True) 
                        if attr['type'] == node_type]
            nx.draw_networkx_nodes(self.graph, pos,
                                 nodelist=node_list,
                                 node_color=color,
                                 node_size=1500)
        
        # Draw edges
        nx.draw_networkx_edges(self.graph, pos, edge_color='gray', arrows=True)
        
        # Add labels
        labels = {node: data['text'][:20] if 'text' in data else data.get('source', '')[:20]
                 for node, data in self.graph.nodes(data=True)}
        nx.draw_networkx_labels(self.graph, pos, labels, font_size=8)
        
        plt.title("Knowledge Graph Visualization")
        plt.axis('off')
        plt.savefig('knowledge_graph.png', bbox_inches='tight', dpi=300)
        plt.close()

class OptimizedAnalyzer:
    def __init__(self, use_gpu: bool = False):
        """Initialize the analyzer with all required components"""
        load_dotenv()
        self.fact_check_api_key = os.getenv('FACT_CHECK_API_KEY')
        self.logger = logging.getLogger('analyzer')
        self.device = 'cuda' if use_gpu and torch.cuda.is_available() else 'cpu'
        
        # API keys
        self.fact_check_api_key = os.getenv('FACT_CHECK_API_KEY')
        self.google_api_key = os.getenv('GOOGLE_API_KEY')
        
        # Initialize components
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
            r"shocking", r"critical", r"emergency", r"life-changing"
        ]
        
        self.STEREOTYPE_PATTERNS = [
            r"all \w+s are", r"\w+ people always", r"typical \w+ behavior",
            r"women can't", r"men are always"
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

    async def fact_check(self, text: str) -> List[Dict]:
        """Perform async fact checking"""
        if not self.fact_check_api_key:
            return []

        cache_key = f"fact_check_{text}"
        if cache_key in self.cache:
            cache_entry = self.cache[cache_key]
            if time() - cache_entry['timestamp'] < CACHE_TTL:
                return cache_entry['data']

        await self._init_session()
        base_url = "https://factchecktools.googleapis.com/v1alpha1/claims:search"
        params = {
            'query': text,
            'key': self.fact_check_api_key
        }

        for attempt in range(MAX_RETRIES):
            try:
                async with self.session.get(base_url, params=params, timeout=TIMEOUT) as response:
                    if response.status == 200:
                        data = await response.json()
                        self.cache[cache_key] = {
                            'data': data.get('claims', []),
                            'timestamp': time()
                        }
                        return data.get('claims', [])
            except Exception as e:
                self.logger.error(f"Fact check attempt {attempt + 1} failed: {str(e)}")
                await asyncio.sleep(1 * (attempt + 1))
        
        return []

    async def analyze_text(self, text: str) -> AnalysisResult:
        """Perform comprehensive text analysis"""
        start_time = time()
        
        try:
            with ThreadPoolExecutor() as executor:
                loop = asyncio.get_event_loop()
                
                # Schedule all ML tasks concurrently
                sentiment_future = loop.run_in_executor(
                    executor,
                    self.sentiment_pipeline,
                    text
                )
                
                ner_future = loop.run_in_executor(
                    executor,
                    self.ner_pipeline,
                    text
                )
                
                classification_future = loop.run_in_executor(
                    executor,
                    self.zero_shot_classifier,
                    text,
                    ["true claim", "false claim"]
                )
                
                pattern_future = loop.run_in_executor(
                    executor,
                    self.detect_patterns,
                    text
                )
                
                fact_check_future = self.fact_check(text)
                
                sentiment, ner_result, classification, patterns, fact_checks = await asyncio.gather(
                    sentiment_future,
                    ner_future,
                    classification_future,
                    pattern_future,
                    fact_check_future
                )
                
                emotional_triggers, stereotypes = patterns
                
                entities = [{
                    'text': entity['word'],
                    'type': entity['entity_group']
                } for entity in ner_result]
                
                # Update knowledge graph
                self.knowledge_graph.add_fact(text, entities, fact_checks, sentiment[0] if sentiment else {})
                
                # Calculate manipulation score
                manipulation_score = self._compute_manipulation_score(
                    sentiment,
                    fact_checks,
                    emotional_triggers,
                    stereotypes
                )
                
                # Optional: Generate AI analysis if available
                generative_analysis = None
                if self.generative_model:
                    try:
                        response = await loop.run_in_executor(
                            executor,
                            self.generative_model.generate_content,
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
                    processing_time=time() - start_time,
                    entities=entities,
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

async def main():
    """Main execution function"""
    analyzer = OptimizedAnalyzer(use_gpu=True)
    
    test_texts = [
        "Breaking news: Major tech company XYZ accused of data breach affecting millions.",
        "Scientific study shows clear evidence of climate change impact.",
        "New startup ABC claims revolutionary AI breakthrough in healthcare."
    ]
    
    try:
        for text in test_texts:
            print(f"\nAnalyzing text: {text}")
            result = await analyzer.analyze_text(text)
            
            print(f"Sentiment: {result.sentiment}")
            print(f"Manipulation Score: {result.manipulation_score}")
            print(f"Entities Found: {len(result.entities)}")
            print(f"Fact Checks: {len(result.fact_checks)}")
            print(f"Processing Time: {result.processing_time:.2f} seconds")
            
            # Visualize knowledge graph after all analyses
            analyzer.knowledge_graph.visualize_graph()
            
            print("-" * 50)
    finally:
        await analyzer._close_session()

if __name__ == "__main__":
    asyncio.run(main())