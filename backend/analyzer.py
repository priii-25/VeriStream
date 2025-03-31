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
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from config import CACHE_TTL, MAX_RETRIES, TIMEOUT, BATCH_SIZE, FACT_CHECK_API_KEY, GOOGLE_API_KEY
from knowledge_graph import KnowledgeGraphManager

# Set up logging with detailed output
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('analyzer')

@dataclass
class AnalysisResult:
    """Data class to store analysis results"""
    text: str
    political_bias: Dict  # Updated from sentiment to political_bias
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
        self.logger.debug(f"Initializing with device: {self.device}")
        self._load_cache()
        self._initialize_ml_pipelines()
        self._setup_detection_patterns()

    def _initialize_ml_pipelines(self):
        """Initialize ML pipelines with optimized batch processing"""
        try:
            self.logger.info("Initializing ML pipelines...")
            # Load fine-tuned political bias model
            model_path = "/Users/kartik/Desktop/vs/VeriStream/backend/models/political-sentiment-model"
            self.logger.debug(f"Loading political bias model from: {model_path}")
            if not os.path.exists(model_path):
                self.logger.error(f"Model path does not exist: {model_path}")
                raise FileNotFoundError(f"Model path does not exist: {model_path}")
            
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.bias_model = AutoModelForSequenceClassification.from_pretrained(model_path)
            self.bias_model.to(self.device)
            self.bias_model.eval()
            self.bias_label_map = {0: "Leftist", 1: "Rightish", 2: "Centric", 3: "Party-Specific"}
            self.logger.debug(f"Political bias model loaded. Labels: {self.bias_label_map}")

            # Keep NER and zero-shot pipelines
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
            r"get rich quick", r"lose weight fast",
        ]
        self.STEREOTYPE_PATTERNS = [
            r"women can't", r"men are always",
            r"liberals are", r"conservatives are",
        ]
        self.logger.debug(f"Emotional triggers: {self.EMOTIONAL_TRIGGERS[:5]}... (total: {len(self.EMOTIONAL_TRIGGERS)})")
        self.logger.debug(f"Stereotype patterns: {self.STEREOTYPE_PATTERNS[:5]}... (total: {len(self.STEREOTYPE_PATTERNS)})")

    def _load_cache(self):
        """Load cache from file with error handling"""
        try:
            if self.cache_file.exists():
                with open(self.cache_file, 'r') as f:
                    self.cache = json.load(f)
                self.logger.debug(f"Cache loaded from {self.cache_file}. Size: {len(self.cache)} entries")
            else:
                self.cache = {}
                self.logger.debug("No cache file found, starting with empty cache")
        except Exception as e:
            self.logger.error(f"Error loading cache: {str(e)}")
            self.cache = {}

    @lru_cache(maxsize=1000)
    def detect_patterns(self, text: str) -> tuple[List[str], List[str]]:
        """Detect emotional triggers and stereotypes with caching"""
        self.logger.debug(f"Detecting patterns in text: {text[:50]}...")
        emotional_triggers = [
            trigger for trigger in self.EMOTIONAL_TRIGGERS
            if re.search(trigger, text, re.IGNORECASE)
        ]
        stereotypes = [
            pattern for pattern in self.STEREOTYPE_PATTERNS
            if re.search(pattern, text, re.IGNORECASE)
        ]
        self.logger.debug(f"Found emotional triggers: {emotional_triggers}")
        self.logger.debug(f"Found stereotypes: {stereotypes}")
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
                self.logger.debug(f"Returning cached fact check for: {text[:50]}...")
                return cache_entry['data']

        base_url = "https://factchecktools.googleapis.com/v1alpha1/claims:search"
        params = {'query': text, 'key': self.fact_check_api_key}
        self.logger.debug(f"Fact checking via API: {text[:50]}...")

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
                        self.logger.debug(f"Fact check result: {data.get('claims', [])}")
                        return data.get('claims', [])
                except Exception as e:
                    self.logger.error(f"Fact check attempt {attempt + 1} failed: {str(e)}")
                    await asyncio.sleep(1 * (attempt + 1))
        return []

    def _analyze_political_bias(self, text: str) -> Dict:
        """Analyze political bias using the fine-tuned model"""
        self.logger.debug(f"Analyzing political bias for text: {text}")
        try:
            inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            self.logger.debug(f"Tokenized inputs: {inputs}")
            with torch.no_grad():
                outputs = self.bias_model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()[0]
            label_idx = probs.argmax()
            result = {
                "label": self.bias_label_map[label_idx],
                "score": float(probs[label_idx]),
                "all_scores": {self.bias_label_map[i]: float(probs[i]) for i in range(4)}
            }
            self.logger.debug(f"Political bias result: {result}")
            return result
        except Exception as e:
            self.logger.error(f"Political bias analysis failed: {str(e)}")
            return {"label": "Unknown", "score": 0.0, "all_scores": {}}

    async def analyze_text(self, text: str) -> AnalysisResult:
        """Perform comprehensive text analysis asynchronously"""
        start_time = time.time()
        self.logger.info(f"Starting analysis for text: {text}")

        try:
            # Run async fact-checking
            fact_checks = await self.fact_check(text)
            
            # Synchronous ML tasks
            political_bias = self._analyze_political_bias(text)
            ner_result = self.ner_pipeline(text)
            classification = self.zero_shot_classifier(text, ["true claim", "false claim"])
            emotional_triggers, stereotypes = self.detect_patterns(text)

            entities = [{
                'text': entity['word'],
                'type': entity['entity_group']
            } for entity in ner_result]
            locations = [entity for entity in entities if entity['type'] == 'LOC']

            self.logger.debug(f"Entities: {entities}")
            self.logger.debug(f"Locations: {locations}")

            self.knowledge_graph.add_fact(
                text, 
                entities, 
                fact_checks, 
                political_bias
            )

            manipulation_score = self._compute_manipulation_score(
                political_bias,
                fact_checks,
                emotional_triggers,
                stereotypes
            )

            result = AnalysisResult(
                text=text,
                political_bias=political_bias,
                fact_checks=fact_checks,
                emotional_triggers=emotional_triggers,
                stereotypes=stereotypes,
                manipulation_score=manipulation_score,
                processing_time=time.time() - start_time,
                entities=entities,
                locations=locations,
                knowledge_graph=self.knowledge_graph.get_graph_data()
            )
            self.logger.info(f"Analysis completed. Political Bias: {result.political_bias}")
            return result

        except Exception as e:
            self.logger.error(f"Analysis failed: {str(e)}")
            raise

    def _compute_manipulation_score(self, political_bias: Dict, fact_checks: List, emotional_triggers: List, stereotypes: List) -> float:
        """Compute manipulation risk score with weighted factors"""
        try:
            bias_weight = 0.3
            fact_weight = 0.3
            trigger_weight = 0.2
            stereotype_weight = 0.2

            bias_score = 1.0 if political_bias.get('label') in ['Leftist', 'Rightish'] and political_bias.get('score', 0) > 0.7 else 0.0
            fact_score = 1.0 if not fact_checks else 0.0
            trigger_score = min(len(emotional_triggers) * 0.2, 1.0)
            stereotype_score = min(len(stereotypes) * 0.2, 1.0)

            final_score = (
                bias_score * bias_weight +
                fact_score * fact_weight +
                trigger_score * trigger_weight +
                stereotype_score * stereotype_weight
            )

            self.logger.debug(f"Manipulation score components - Bias: {bias_score}, Fact: {fact_score}, Trigger: {trigger_score}, Stereotype: {stereotype_score}")
            self.logger.debug(f"Final manipulation score: {final_score}")
            return round(final_score, 2)
        except Exception as e:
            self.logger.error(f"Manipulation score computation failed: {e}")
            return 0.5

# For testing standalone with enhanced debugging
if __name__ == "__main__":
    print("Starting analyzer test...")
    logging.basicConfig(level=logging.DEBUG, force=True)
    
    async def test():
        try:
            print("Initializing analyzer...")
            analyzer = OptimizedAnalyzer(use_gpu=True)
            print("Analyzer initialized, analyzing text...")
            test_text = "COVID-19 vaccines are ineffective"
            result = await analyzer.analyze_text(test_text)
            print("Analysis complete, result:")
            print(f"Text: {result.text}")
            print(f"Political Bias: {result.political_bias}")
            print(f"Fact Checks: {result.fact_checks}")
            print(f"Emotional Triggers: {result.emotional_triggers}")
            print(f"Stereotypes: {result.stereotypes}")
            print(f"Manipulation Score: {result.manipulation_score}")
            print(f"Processing Time: {result.processing_time}")
            print(f"Entities: {result.entities}")
            print(f"Locations: {result.locations}")
            print(f"Knowledge Graph: {result.knowledge_graph}")
        except Exception as e:
            print(f"Error occurred: {str(e)}")
            logger.error(f"Test failed: {str(e)}")

    asyncio.run(test())
    print("Test finished.")