import spacy
from transformers import pipeline
import requests
from collections import deque
import time
import hashlib
import logging
import threading
from textblob import TextBlob
from typing import Dict, List, Tuple

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

nlp = spacy.load("en_core_web_sm")
classifier = pipeline("text-classification", model="distilbert-base-uncased")  # Remove top_k=None

class FactCheckManager:
    def __init__(self, api_key: str, max_calls_per_hour=1000, cooldown=5):
        self.api_key = api_key
        self.queue = deque()
        self.cache = set()
        self.lock = threading.Lock()
        self.call_count = 0
        self.max_calls = max_calls_per_hour
        self.cooldown = cooldown
        self.last_call_time = 0
        self.priority_threshold = 0.7
        self.results = {}
        self.running = True
        self.thread = threading.Thread(target=self.process_queue, daemon=True)
        self.thread.start()

    def real_fact_check_api(self, claims: List[str]) -> List[Dict]:
        """Use Google Fact Check Tools API."""
        try:
            url = "https://factchecktools.googleapis.com/v1alpha1/claims:search"
            results = []
            for claim in claims:
                params = {"query": claim, "key": self.api_key}
                response = requests.get(url, params=params)
                if response.status_code == 200:
                    data = response.json()
                    if data.get("claims"):
                        claim_review = data["claims"][0]["claimReview"][0]
                        result = {
                            "verdict": claim_review.get("textualRating", "Unknown"),
                            "evidence": f"{claim_review.get('publisher', {}).get('name', 'Unknown')} - {claim_review.get('url', '')}"
                        }
                    else:
                        result = {"verdict": "Unknown", "evidence": "No matching fact-check found"}
                else:
                    result = {"verdict": "Error", "evidence": f"API error: {response.text}"}
                results.append(result)
            logger.info(f"Fact-checked {len(claims)} claims: {results}")
            return results
        except Exception as e:
            logger.error(f"Fact-check API call failed: {e}")
            return [{"verdict": "Error", "evidence": "API unavailable"}] * len(claims)

    def add_to_queue(self, prioritized_claims):
        with self.lock:
            for claim, priority in prioritized_claims:
                self.queue.append((claim, priority))
            logger.info(f"Added {len(prioritized_claims)} claims to queue. Queue size: {len(self.queue)}")

    def adjust_threshold(self):
        if self.call_count > self.max_calls * 0.8:
            self.priority_threshold = min(0.9, self.priority_threshold + 0.1)
            logger.info(f"Adjusted threshold to {self.priority_threshold} due to high API usage")

    def process_queue(self):
        while self.running:
            with self.lock:
                if self.call_count >= self.max_calls:
                    logger.warning("API call limit reached")
                    time.sleep(3600 // self.max_calls)
                    self.call_count = 0
                
                current_time = time.time()
                if current_time - self.last_call_time < self.cooldown or not self.queue:
                    time.sleep(1)
                    continue
                
                self.adjust_threshold()
                batch = []
                while self.queue and len(batch) < 2:
                    claim, priority = self.queue[0]
                    if priority >= self.priority_threshold:
                        batch.append(self.queue.popleft()[0])
                        self.cache.add(hashlib.md5(claim.encode()).hexdigest())
                    else:
                        self.queue.popleft()
                
                if batch:
                    self.call_count += 1
                    results = self.real_fact_check_api(batch)
                    self.last_call_time = current_time
                    for claim, result in zip(batch, results):
                        self.results[claim] = result
                        logger.info(f"Claim: '{claim}' → Verdict: {result['verdict']}, Evidence: {result['evidence']}")

    def stop(self):
        self.running = False
        self.thread.join()

    def get_results(self, claims: List[str], timeout: int = 10) -> List[Dict]:
        start_time = time.time()
        results = []
        while time.time() - start_time < timeout:
            with self.lock:
                results = [self.results.get(claim, {"verdict": "Pending", "evidence": "Processing"}) 
                          for claim in claims]
                if all(r["verdict"] != "Pending" for r in results):
                    break
            time.sleep(1)
        return results

def segment_sentences(text: str) -> List[str]:
    try:
        doc = nlp(text)
        sentences = [sent.text.strip() for sent in doc.sents if len(sent.text.strip()) > 5]
        logger.info(f"Segmented {len(sentences)} sentences")
        return sentences
    except Exception as e:
        logger.error(f"Segmentation failed: {e}")
        return []

def classify_factual(sentences: List[str]) -> List[Tuple[str, float]]:
    factual_sentences = []
    for sent in sentences:
        try:
            results = classifier(sent)
            top_result = max(results, key=lambda x: x['score'])  # Get highest scoring label
            label, score = top_result['label'], top_result['score']
            if label == "POSITIVE" and score > 0.85:
                factual_sentences.append((sent, score))
        except Exception as e:
            logger.error(f"Classification failed for '{sent}': {e}")
    return factual_sentences

def prioritize_claims(factual_sentences: List[Tuple[str, float]], cache: set) -> List[Tuple[str, float]]:
    prioritized = []
    for sent, factual_score in factual_sentences:
        try:
            doc = nlp(sent)
            specificity = 0.9 if any(ent.label_ in ["PERSON", "ORG", "DATE", "GPE", "QUANTITY"] for ent in doc.ents) else 0.7
            blob = TextBlob(sent)
            sentiment_score = abs(blob.sentiment.polarity)
            has_strong_assertion = any(token.text.lower() in ["is", "are", "always", "never"] for token in doc)
            controversy = 1.0 if (sentiment_score > 0.3 and has_strong_assertion) else 0.5
            sent_hash = hashlib.md5(sent.encode()).hexdigest()
            novelty = 0.5 if sent_hash in cache else 0.8
            priority = (0.4 * specificity) + (0.4 * controversy) + (0.2 * novelty)
            if priority > 0.7:
                prioritized.append((sent, priority))
        except Exception as e:
            logger.error(f"Prioritization failed for '{sent}': {e}")
    return prioritized

def fact_check_text(text: str, manager: FactCheckManager) -> List[Dict]:
    sentences = segment_sentences(text)
    factual_sentences = classify_factual(sentences)
    prioritized_claims = prioritize_claims(factual_sentences, manager.cache)
    manager.add_to_queue(prioritized_claims)
    claims = [claim for claim, _ in prioritized_claims]
    return manager.get_results(claims)

if __name__ == "__main__":
    manager = FactCheckManager(api_key="YOUR_GOOGLE_API_KEY_HERE")
    text = "The Earth is flat and the moon is made of cheese."
    results = fact_check_text(text, manager)
    print(results)
    time.sleep(5)
    manager.stop()