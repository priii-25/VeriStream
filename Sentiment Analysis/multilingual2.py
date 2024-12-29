import os
import re
import json
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import requests
import torch
import google.generativeai as genai
from dotenv import load_dotenv
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer

# Custom exceptions
class APIKeyError(Exception):
    """Raised when required API keys are missing."""
    pass

class ModelInitializationError(Exception):
    """Raised when ML models fail to initialize."""
    pass

# Enums for better type safety
class SentimentLabel(str, Enum):
    NEGATIVE = 'LABEL_0'
    NEUTRAL = 'LABEL_1'
    POSITIVE = 'LABEL_2'

@dataclass
class AnalysisResult:
    """Structured container for analysis results."""
    text: str
    transformer_sentiment: List[Dict]
    generative_sentiment: Dict[str, str]
    fact_check_results: List[Dict]
    emotional_triggers: List[str]
    stereotypes: List[str]
    manipulation_risk_score: float
    language: str

class MultilingualAnalyzer:
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the analyzer with optional configuration file.
        
        Args:
            config_path: Path to configuration file (optional)
        """
        self._load_environment(config_path)
        self._initialize_logging()
        self._validate_api_keys()
        self._initialize_components()

    def _load_environment(self, config_path: Optional[str]) -> None:
        """Load environment variables from .env file or specified config."""
        if config_path and os.path.exists(config_path):
            load_dotenv(config_path)
        else:
            load_dotenv()
        
        self.google_api_key = os.getenv('GOOGLE_API_KEY')
        self.fact_check_api_key = os.getenv('FACT_CHECK_API_KEY')

    def _initialize_logging(self) -> None:
        """Configure logging with rotating file handler."""
        self.logger = logging.getLogger(__name__)
        if not self.logger.handlers:
            logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                handlers=[
                    logging.FileHandler('multilingual_analyzer.log'),
                    logging.StreamHandler()
                ]
            )

    def _validate_api_keys(self) -> None:
        """Validate required API keys are present."""
        missing_keys = []
        if not self.google_api_key:
            missing_keys.append('GOOGLE_API_KEY')
        if not self.fact_check_api_key:
            missing_keys.append('FACT_CHECK_API_KEY')
        
        if missing_keys:
            self.logger.warning(f"Missing API keys: {', '.join(missing_keys)}")

    def _initialize_components(self) -> None:
        """Initialize all analysis components."""
        try:
            self._setup_sentiment_analysis()
            self._setup_detection_patterns()
            if self.google_api_key:
                self._setup_generative_ai()
        except Exception as e:
            raise ModelInitializationError(f"Failed to initialize components: {str(e)}")

    def _setup_sentiment_analysis(self) -> None:
        """Initialize sentiment analysis pipeline."""
        try:
            # Using cardiffnlp/twitter-roberta-base-sentiment for sentiment analysis
            model_name = "cardiffnlp/twitter-roberta-base-sentiment"
            self.sentiment_pipeline = pipeline(
                "sentiment-analysis",
                model=model_name,
                device="mps" if torch.backends.mps.is_available() else "cpu"
            )
            self.logger.info("Sentiment analysis pipeline initialized successfully")
        except Exception as e:
            self.logger.error(f"Sentiment analysis setup failed: {e}")
            raise ModelInitializationError(str(e))

    def _setup_detection_patterns(self) -> None:
        """Setup emotional triggers and stereotype patterns."""
        self.EMOTIONAL_TRIGGERS = [
            r"breaking news", r"fear of", r"unprecedented", r"urgent", 
            r"shocking", r"critical", r"emergency", r"life-changing",
            r"crisis", r"disaster", r"terrifying", r"catastrophic"
        ]
        
        self.STEREOTYPE_PATTERNS = [
            r"all \w+s are", r"\w+ people always", r"typical \w+ behavior", 
            r"women can't", r"men are always", r"they're all like that",
            r"those people", r"that group"
        ]

    def _setup_generative_ai(self) -> None:
        """Configure Google Generative AI."""
        try:
            genai.configure(api_key=self.google_api_key)
            self.generative_model = genai.GenerativeModel('gemini-pro')
            self.logger.info("Generative AI initialized successfully")
        except Exception as e:
            self.logger.error(f"Generative AI setup failed: {e}")
            self.generative_model = None

    def detect_emotional_triggers(self, text: str) -> List[str]:
        """
        Detect emotionally manipulative phrases in text.
        
        Args:
            text: Input text to analyze
            
        Returns:
            List of detected trigger phrases
        """
        return [
            trigger for trigger in self.EMOTIONAL_TRIGGERS 
            if re.search(trigger, text.lower(), re.IGNORECASE)
        ]

    def detect_stereotypes(self, text: str) -> List[str]:
        """
        Detect stereotypical patterns in text.
        
        Args:
            text: Input text to analyze
            
        Returns:
            List of detected stereotype patterns
        """
        return [
            pattern for pattern in self.STEREOTYPE_PATTERNS 
            if re.search(pattern, text.lower(), re.IGNORECASE)
        ]

    def perform_fact_check(self, query: str) -> List[Dict]:
        """
        Perform fact-checking using Google Fact Check API.
        
        Args:
            query: Text to fact check
            
        Returns:
            List of fact check results
        """
        if not self.fact_check_api_key:
            self.logger.warning("No fact-check API key available")
            return []
        
        try:
            base_url = "https://factchecktools.googleapis.com/v1alpha1/claims:search"
            params = {
                'query': query,
                'key': self.fact_check_api_key,
                'languageCode': 'en'  # Default to English
            }
            
            response = requests.get(base_url, params=params)
            response.raise_for_status()
            
            data = response.json()
            fact_check_results = data.get('claims', [])
            
            return [
                {
                    'claim': claim.get('text', ''),
                    'rating': claim.get('claimReview', [{}])[0].get('textualRating', ''),
                    'publisher': claim.get('claimReview', [{}])[0].get('publisher', {}).get('name', ''),
                    'url': claim.get('claimReview', [{}])[0].get('url', '')
                } for claim in fact_check_results
            ]
        
        except requests.RequestException as e:
            self.logger.error(f"Fact-check API request failed: {e}")
            return []

    def generative_sentiment_analysis(self, text: str) -> Dict[str, Any]:
        """
        Perform sentiment analysis using Generative AI.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary containing sentiment analysis results
        """
        if not self.generative_model:
            return {
                "sentiment": "Unknown",
                "explanation": "Generative AI model not available"
            }
        
        try:
            prompt = f"""
            Analyze the sentiment of the following text and provide:
            1. Overall sentiment (Positive/Negative/Neutral)
            2. A brief explanation of the sentiment
            3. Any potential bias or emotional manipulation indicators
            4. Key themes or topics discussed

            Text: {text}
            """
            
            response = self.generative_model.generate_content(prompt)
            return {
                "sentiment": "Analyzed",
                "explanation": response.text
            }
        except Exception as e:
            self.logger.error(f"Generative AI analysis error: {e}")
            return {
                "sentiment": "Error",
                "explanation": str(e)
            }

    def compute_manipulation_score(self, 
                                sentiment_results: List[Dict], 
                                fact_check_results: List, 
                                emotional_triggers: List, 
                                stereotypes: List) -> float:
        """
        Compute manipulation risk score based on various factors.
        
        Returns:
            Float between 0 and 1, where higher values indicate higher risk
        """
        try:
            # Extract the highest confidence sentiment
            sentiment_score = max(
                result['score'] for result in sentiment_results[0]
            )
            
            # Calculate component scores
            fact_check_score = min(len(fact_check_results) * 0.2, 1.0)
            trigger_score = min(len(emotional_triggers) * 0.3, 1.0)
            stereotype_score = min(len(stereotypes) * 0.4, 1.0)
            
            # Weighted average of components
            weights = {
                'sentiment': 0.3,
                'fact_check': 0.3,
                'triggers': 0.2,
                'stereotypes': 0.2
            }
            
            final_score = (
                (sentiment_score * weights['sentiment']) +
                (fact_check_score * weights['fact_check']) +
                (trigger_score * weights['triggers']) +
                (stereotype_score * weights['stereotypes'])
            )
            
            return round(final_score, 2)
            
        except Exception as e:
            self.logger.error(f"Manipulation score computation error: {e}")
            return 0.5

    def analyze_text(self, text: str) -> AnalysisResult:
        """
        Perform comprehensive text analysis.
        
        Args:
            text: Input text to analyze
            
        Returns:
            AnalysisResult object containing all analysis results
        """
        try:
            self.logger.info(f"Starting analysis of text: {text[:100]}...")
            
            # Perform all analyses
            sentiment_results = self.sentiment_pipeline(text)
            generative_analysis = self.generative_sentiment_analysis(text)
            fact_check_results = self.perform_fact_check(text)
            emotional_triggers = self.detect_emotional_triggers(text)
            stereotypes = self.detect_stereotypes(text)
            
            # Compute manipulation score
            manipulation_score = self.compute_manipulation_score(
                sentiment_results,
                fact_check_results,
                emotional_triggers,
                stereotypes
            )
            
            # Create result object
            result = AnalysisResult(
                text=text,
                transformer_sentiment=sentiment_results,
                generative_sentiment=generative_analysis,
                fact_check_results=fact_check_results,
                emotional_triggers=emotional_triggers,
                stereotypes=stereotypes,
                manipulation_risk_score=manipulation_score,
                language="auto-detected"  # You can add language detection if needed
            )
            
            self.logger.info(f"Analysis completed successfully for text: {text[:50]}...")
            return result

        except Exception as e:
            self.logger.error(f"Analysis failed: {e}")
            raise

    def batch_analyze(self, texts: List[str], batch_size: int = 5) -> List[AnalysisResult]:
        """
        Analyze multiple texts in batches.
        
        Args:
            texts: List of texts to analyze
            batch_size: Number of texts to process in parallel
            
        Returns:
            List of AnalysisResult objects
        """
        results = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            self.logger.info(f"Processing batch {i//batch_size + 1}")
            batch_results = [self.analyze_text(text) for text in batch]
            results.extend(batch_results)
        return results

def main():
    """Main execution function."""
    try:
        # Initialize analyzer
        analyzer = MultilingualAnalyzer()
        
        # Test texts in different languages
        test_texts = [
            "मैग्नस कार्लसन ने गुकेश के साथ शतरंज खिताबी मुकाबले से इनकार किया: अब इस सर्कस का हिस्सा नहीं",
            "World Chess Champion D Gukesh gets a 'message' from Elon Musk",
            "Visa H-1B: Propuesta de revisión importante del programa de visas en la Casa Blanca ahora."
        ]
        
        # Process texts
        results = analyzer.batch_analyze(test_texts)
        
        # Save results
        output_file = 'analysis_results.json'
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(
                [vars(result) for result in results],
                f,
                ensure_ascii=False,
                indent=2
            )
            
        print(f"Analysis complete. Results saved to {output_file}")
            
    except Exception as e:
        logging.error(f"Main execution failed: {e}")
        raise

if __name__ == "__main__":
    main()