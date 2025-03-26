# backend/analyzer.py
import os
import re
import time
import json
import logging
import aiohttp
import torch
import asyncio
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass
from functools import lru_cache
from transformers import pipeline
from config import CACHE_TTL, MAX_RETRIES, TIMEOUT, BATCH_SIZE, FACT_CHECK_API_KEY, GOOGLE_API_KEY
from knowledge_graph import KnowledgeGraphManager

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('analyzer')

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

class OptimizedAnalyzer:
    def __init__(self, use_gpu: bool = False):
        """Initialize the analyzer with all required components"""
        self.fact_check_api_key = FACT_CHECK_API_KEY
        self.logger = logger
        self.device = 'cuda' if use_gpu and torch.cuda.is_available() else 'cpu'
        self.google_api_key = GOOGLE_API_KEY
        self.cache_file = Path('analysis_cache.json')
        self.knowledge_graph = KnowledgeGraphManager()
        self._load_cache()
        self._initialize_ml_pipelines()
        self._setup_detection_patterns()

    def _initialize_ml_pipelines(self):
        """Initialize ML pipelines with optimized batch processing"""
        try:
            self.logger.info("Initializing ML pipelines...")
            self.sentiment_pipeline = pipeline(
                "sentiment-analysis",
                model="cardiffnlp/twitter-roberta-base-sentiment",
                device=0 if self.device == "cuda" else -1,  # Fixed device mapping
                batch_size=BATCH_SIZE
            )
            self.ner_pipeline = pipeline(
                task="ner",
                model="dslim/bert-base-NER",
                aggregation_strategy="simple",
                device=0 if self.device == "cuda" else -1,
                batch_size=BATCH_SIZE
            )
            self.zero_shot_classifier = pipeline(
                task="zero-shot-classification",
                model="facebook/bart-large-mnli",
                device=0 if self.device == "cuda" else -1,
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
        """Perform async fact checking with aiohttp"""
        if not self.fact_check_api_key:
            self.logger.warning("Fact check API key is missing")
            return []

        cache_key = f"fact_check_{text}"
        if cache_key in self.cache:
            cache_entry = self.cache[cache_key]
            if time.time() - cache_entry['timestamp'] < CACHE_TTL:
                return cache_entry['data']

        base_url = "https://factchecktools.googleapis.com/v1alpha1/claims:search"
        params = {'query': text, 'key': self.fact_check_api_key}

        async with aiohttp.ClientSession() as session:
            for attempt in range(MAX_RETRIES):
                try:
                    async with session.get(base_url, params=params, timeout=aiohttp.ClientTimeout(total=TIMEOUT)) as response:
                        response.raise_for_status()
                        data = await response.json()
                        self.cache[cache_key] = {
                            'data': data.get('claims', []),
                            'timestamp': time.time()
                        }
                        with open(self.cache_file, 'w') as f:
                            json.dump(self.cache, f)
                        return data.get('claims', [])
                except Exception as e:
                    self.logger.error(f"Fact check attempt {attempt + 1} failed: {str(e)}")
                    await asyncio.sleep(1 * (attempt + 1))
        return []

    async def analyze_text(self, text: str) -> AnalysisResult:
        """Perform comprehensive text analysis asynchronously"""
        start_time = time.time()

        try:
            # Run async fact-checking
            fact_checks = await self.fact_check(text)
            
            # Synchronous ML tasks (these are fast enough to run synchronously)
            sentiment = self.sentiment_pipeline(text)
            ner_result = self.ner_pipeline(text)
            classification = self.zero_shot_classifier(text, ["true claim", "false claim"])
            emotional_triggers, stereotypes = self.detect_patterns(text)

            entities = [{
                'text': entity['word'],
                'type': entity['entity_group']
            } for entity in ner_result]

            locations = [entity for entity in entities if entity['type'] == 'LOC']

            self.knowledge_graph.add_fact(
                text, 
                entities, 
                fact_checks, 
                sentiment[0] if sentiment else {}
            )

            manipulation_score = self._compute_manipulation_score(
                sentiment,
                fact_checks,
                emotional_triggers,
                stereotypes
            )

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
                knowledge_graph=self.knowledge_graph.get_graph_data()
            )

        except Exception as e:
            self.logger.error(f"Analysis failed: {str(e)}")
            raise

    def _compute_manipulation_score(self, sentiment: List[Dict], fact_checks: List, emotional_triggers: List, stereotypes: List) -> float:
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

# For testing standalone
if __name__ == "__main__":
    async def test():
        analyzer = OptimizedAnalyzer(use_gpu=True)
        result = await analyzer.analyze_text("Breaking news: This is a shocking revelation!")
        print(result)

    asyncio.run(test())