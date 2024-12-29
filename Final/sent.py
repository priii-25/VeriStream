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
    generative_analysis: Optional[Dict] = None

class OptimizedAnalyzer:
    def __init__(self, use_gpu: bool = False):
        """Initialize the analyzer with all required components"""
        load_dotenv()
        print(f"Env vars loaded - FACT_CHECK_API_KEY present: {'FACT_CHECK_API_KEY' in os.environ}")
        self.fact_check_api_key = os.getenv('FACT_CHECK_API_KEY')
        print(f"Loaded API key value exists: {bool(self.fact_check_api_key)}")
        self.logger = logging.getLogger('analyzer')
        self.device = 'cuda' if use_gpu and torch.cuda.is_available() else 'cpu'
        
        # API keys
        self.fact_check_api_key = os.getenv('FACT_CHECK_API_KEY')
        self.google_api_key = os.getenv('GOOGLE_API_KEY')
        
        # Initialize components
        self.session = None
        self.cache_file = Path('analysis_cache.json')
        self._load_cache()
        self._initialize_ml_pipelines()
        self._setup_detection_patterns()
        self._setup_generative_ai()

    def _initialize_ml_pipelines(self):
        """Initialize all ML pipelines with optimized batch processing"""
        try:
            self.logger.info("Initializing ML pipelines...")
            
            # Sentiment analysis pipeline
            self.sentiment_pipeline = pipeline(
                "sentiment-analysis",
                model="cardiffnlp/twitter-roberta-base-sentiment",
                device=self.device,
                batch_size=BATCH_SIZE
            )
            
            # NER pipeline
            self.ner_pipeline = pipeline(
                task="ner",
                model="dslim/bert-base-NER",
                aggregation_strategy="simple",
                device=self.device,
                batch_size=BATCH_SIZE
            )
            
            # Zero-shot classification pipeline
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
        print("Starting fact check...") 
        logger.info("Starting fact check") 

        print(f"API key value: {bool(self.fact_check_api_key)}")
    
        if not self.fact_check_api_key:
            self.logger.warning("No fact check API key found")
            print("No fact check API key found")
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

        print(f"Making API request to: {base_url}")

        for attempt in range(MAX_RETRIES):
            try:
                print(f"Attempt {attempt + 1} of {MAX_RETRIES}")  # Add this
                async with self.session.get(base_url, params=params, timeout=TIMEOUT) as response:
                    print(f"Response status: {response.status}")  # Add this
                    if response.status == 200:
                        data = await response.json()
                        print(f"API Response: {data}")  # Add this
                        self.cache[cache_key] = {
                            'data': data.get('claims', []),
                            'timestamp': time()
                        }
                        return data.get('claims', [])
                    else:
                        print(f"Error response: {await response.text()}")  # Add this
            except Exception as e:
                print(f"Error during attempt {attempt + 1}: {str(e)}")  # Add this
                self.logger.error(f"Fact check attempt {attempt + 1} failed: {str(e)}")
                await asyncio.sleep(1 * (attempt + 1))
        
        return []

    async def analyze_text(self, text: str) -> AnalysisResult:
        """Perform comprehensive text analysis"""
        start_time = time()
        
        try:
            # Create thread pool for CPU-bound tasks
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
                
                # Pattern detection can be done in parallel
                pattern_future = loop.run_in_executor(
                    executor,
                    self.detect_patterns,
                    text
                )
                
                # Perform fact checking asynchronously
                fact_check_future = self.fact_check(text)
                
                # Wait for all tasks to complete
                sentiment, ner_result, classification, patterns, fact_checks = await asyncio.gather(
                    sentiment_future,
                    ner_future,
                    classification_future,
                    pattern_future,
                    fact_check_future
                )
                
                emotional_triggers, stereotypes = patterns
                
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
                    entities=[{
                        'text': entity['word'],
                        'type': entity['entity_group']
                    } for entity in ner_result],
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
            # Weight factors
            sentiment_weight = 0.3
            fact_weight = 0.3
            trigger_weight = 0.2
            stereotype_weight = 0.2
            
            # Calculate individual scores
            sentiment_score = 1.0 if sentiment and sentiment[0]['label'] == 'NEGATIVE' else 0.0
            fact_score = 1.0 if not fact_checks else 0.0
            trigger_score = min(len(emotional_triggers) * 0.2, 1.0)
            stereotype_score = min(len(stereotypes) * 0.2, 1.0)
            
            # Compute weighted average
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
        "COVID-19 vaccines are ineffective."
    ]
    
    try:
        for text in test_texts:
            result = await analyzer.analyze_text(text)
            print(f"\nAnalysis for: {text}")
            print(f"Sentiment: {result.sentiment}")
            print(f"Manipulation Score: {result.manipulation_score}")
            print(f"Processing Time: {result.processing_time:.2f} seconds")
            print("-" * 50)
    finally:
        await analyzer._close_session()

if __name__ == "__main__":
    asyncio.run(main())