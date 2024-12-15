import os
import re
import json
import logging
from typing import Dict, List, Any
import requests
import google.generativeai as genai
from dotenv import load_dotenv
from transformers import pipeline

logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)

class MultilingualAnalyzer:
    def __init__(self):
        load_dotenv()

        self.google_api_key = os.getenv('GOOGLE_API_KEY')
        self.fact_check_api_key = os.getenv('FACT_CHECK_API_KEY')
        self._setup_sentiment_analysis()
        self._setup_detection_patterns()
        if self.google_api_key:
            self._setup_generative_ai()
        
    def _setup_sentiment_analysis(self):
        """Initialize sentiment analysis pipeline."""
        try:
            self.sentiment_pipeline = pipeline(
                "sentiment-analysis", 
                model="cardiffnlp/twitter-roberta-base-sentiment",
                top_k=None
            )
        except Exception as e:
            logger.error(f"Sentiment analysis setup failed: {e}")
            self.sentiment_pipeline = None

    def _setup_generative_ai(self):
        """Configure Google Generative AI."""
        try:
            genai.configure(api_key=self.google_api_key)
            self.generative_model = genai.GenerativeModel('gemini-pro')
        except Exception as e:
            logger.error(f"Generative AI setup failed: {e}")
            self.generative_model = None

    def _setup_detection_patterns(self):
        """Setup emotional triggers and stereotype patterns."""
        self.EMOTIONAL_TRIGGERS = [
            r"breaking news", r"fear of", r"unprecedented", r"urgent", 
            r"shocking", r"critical", r"emergency", r"life-changing"
        ]
        
        self.STEREOTYPE_PATTERNS = [
            r"all \w+s are", r"\w+ people always", r"typical \w+ behavior", 
            r"women can't", r"men are always"
        ]

    def detect_emotional_triggers(self, text: str) -> List[str]:
        """Detect emotionally manipulative phrases."""
        return [
            trigger for trigger in self.EMOTIONAL_TRIGGERS 
            if re.search(trigger, text, re.IGNORECASE)
        ]

    def detect_stereotypes(self, text: str) -> List[str]:
        """Detect stereotypes in the text."""
        return [
            pattern for pattern in self.STEREOTYPE_PATTERNS 
            if re.search(pattern, text, re.IGNORECASE)
        ]

    def perform_fact_check(self, query: str) -> List[Dict]:
        """
        Perform fact-checking using Google Fact Check API.
        """
        if not self.fact_check_api_key:
            logger.warning("No fact-check API key available.")
            return []
        
        try:
            base_url = "https://factchecktools.googleapis.com/v1alpha1/claims:search"
            params = {
                'query': query,
                'key': self.fact_check_api_key
            }
            
            response = requests.get(base_url, params=params)
            response.raise_for_status()
            
            data = response.json()
            fact_check_results = data.get('claims', [])
            
            return [
                {
                    'claim': claim.get('text', ''),
                    'rating': claim.get('claimReview', [{}])[0].get('textualRating', ''),
                    'publisher': claim.get('claimReview', [{}])[0].get('publisher', {}).get('name', '')
                } for claim in fact_check_results
            ]
        
        except requests.RequestException as e:
            logger.error(f"Fact-check API request failed: {e}")
            return []

    def generative_sentiment_analysis(self, text: str) -> Dict[str, Any]:
        """Perform sentiment analysis using Generative AI."""
        if not self.generative_model:
            return {"sentiment": "Unknown", "explanation": "No generative model available"}
        
        try:
            prompt = f"""
            Analyze the sentiment of the following text. 
            Provide:
            1. Overall sentiment (Positive/Negative/Neutral)
            2. A brief explanation
            3. Potential misinformation indicators

            Text: {text}
            """
            
            response = self.generative_model.generate_content(prompt)
            return {
                "sentiment": "Pending analysis",
                "explanation": response.text
            }
        except Exception as e:
            logger.error(f"Generative AI analysis error: {e}")
            return {"sentiment": "Error", "explanation": str(e)}

    def compute_manipulation_score(self, 
                                   sentiment_results: List[Dict], 
                                   fact_check_results: List, 
                                   emotional_triggers: List, 
                                   stereotypes: List) -> float:
        """Compute manipulation risk score."""
        try:
            dominant_sentiment = max(sentiment_results[0], key=lambda x: x['score'])
            sentiment_map = {
                'LABEL_0': 1.0,  
                'LABEL_1': 0.5,  
                'LABEL_2': 0.0   
            }
            
            sentiment_score = sentiment_map.get(dominant_sentiment['label'], 0.5)
            fact_score = 1.0 if not fact_check_results else 0.5
            trigger_score = 1.0 if not emotional_triggers else 0.5
            stereotype_score = 1.0 if not stereotypes else 0.5

            return round((sentiment_score + fact_score + trigger_score + stereotype_score) / 4, 2)
        except Exception as e:
            logger.error(f"Manipulation score computation error: {e}")
            return 0.5

    def analyze_text(self, text: str) -> Dict[str, Any]:
        """Comprehensive text analysis method."""
        try:
            sentiment_results = self.sentiment_pipeline(text)
            generative_analysis = self.generative_sentiment_analysis(text)
            fact_check_results = self.perform_fact_check(text)
            emotional_triggers = self.detect_emotional_triggers(text)
            stereotypes = self.detect_stereotypes(text)
            manipulation_score = self.compute_manipulation_score(
                sentiment_results, 
                fact_check_results, 
                emotional_triggers, 
                stereotypes
            )
            
            return {
                "text": text,
                "transformer_sentiment": sentiment_results,
                "generative_sentiment": generative_analysis,
                "fact_check_results": fact_check_results,
                "emotional_triggers": emotional_triggers,
                "stereotypes": stereotypes,
                "manipulation_risk_score": manipulation_score
            }
        
        except Exception as e:
            logger.error(f"Comprehensive analysis failed: {e}")
            return {"error": str(e)}

def main():
    """Main execution method."""
    if not os.getenv('GOOGLE_API_KEY'):
        print("Warning: GOOGLE_API_KEY not set in environment variables.")
    
    if not os.getenv('FACT_CHECK_API_KEY'):
        print("Warning: FACT_CHECK_API_KEY not set in environment variables.")
    
    analyzer = MultilingualAnalyzer()
    
    test_texts = [
        "मैग्नस कार्लसन ने गुकेश के साथ शतरंज खिताबी मुकाबले से इनकार किया: अब इस सर्कस का हिस्सा नहीं", 
        "World Chess Champion D Gukesh gets a 'message' from Elon Musk",  
        "Visa H-1B: Propuesta de revisión importante del programa de visas en la Casa Blanca ahora."  
    ]
    
    for text in test_texts:
        print(f"\nAnalyzing text: {text}")
        result = analyzer.analyze_text(text)
        
        print(json.dumps(result, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
    print(f"Google API Key: {os.getenv('GOOGLE_API_KEY')}")
    print(f"Fact Check API Key: {os.getenv('FACT_CHECK_API_KEY')}")