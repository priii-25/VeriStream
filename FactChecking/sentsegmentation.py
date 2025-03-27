import spacy
from transformers import pipeline
import requests
from collections import deque
import time
import hashlib
import logging
import threading
from textblob import TextBlob  # For sentiment analysis as a controversy proxy

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

nlp = spacy.load("en_core_web_sm")
classifier = pipeline("text-classification", model="distilbert-base-uncased", top_k=None) 

def real_fact_check_api(claims, api_key="your_api_key_here"):
    try:
        # Placeholder )
        mock_responses = {
            "The Earth is flat": ("False", "Earth is an oblate spheroid (NASA, 2023)"),
            "The moon is made of cheese": ("False", "Moon is rock and dust (Apollo missions)"),
            "Climate change is a hoax": ("False", "IPCC reports confirm warming trends"),
        }
        results = [mock_responses.get(claim, ("Unknown", "Web search needed")) for claim in claims]
        logger.info(f"API call successful for claims: {claims}")
        return results
    except Exception as e:
        logger.error(f"API call failed: {e}")
        return [("Error", "API unavailable")] * len(claims)

# Step 1: Sentence Segmentation
def segment_sentences(text):
    try:
        doc = nlp(text)
        sentences = [sent.text.strip() for sent in doc.sents if len(sent.text.strip()) > 5]
        logger.info(f"Segmented {len(sentences)} sentences")
        return sentences
    except Exception as e:
        logger.error(f"Segmentation failed: {e}")
        return []

# Step 2: Factual Statement Classification
def classify_factual(sentences):
    factual_sentences = []
    for sent in sentences:
        try:
            result = classifier(sent)[0]
            label, score = result['label'], result['score']
            is_factual = label == "POSITIVE" and score > 0.85
            reason = "Factual" if is_factual else ("Opinion/Question/Noise" if score < 0.5 else "Uncertain")
            
            logger.debug(f"Sentence: '{sent}' → {reason} (Score: {score:.2f})")
            if is_factual:
                factual_sentences.append((sent, score))
        except Exception as e:
            logger.error(f"Classification failed for '{sent}': {e}")
    return factual_sentences

# Step 3: Claim Prioritization (Generalized Controversy Detection)
def prioritize_claims(factual_sentences, cache):
    prioritized = []
    for sent, factual_score in factual_sentences:
        try:
            doc = nlp(sent)
            # Specificity: Presence of named entities or measurable terms
            specificity = 0.9 if any(ent.label_ in ["PERSON", "ORG", "DATE", "GPE", "QUANTITY"] for ent in doc.ents) else 0.7
            
            # Controversy: Use sentiment polarity + strong assertions as a proxy
            blob = TextBlob(sent)
            sentiment_score = abs(blob.sentiment.polarity)  # High polarity = controversial
            has_strong_assertion = any(token.text.lower() in ["is", "are", "always", "never"] for token in doc)
            controversy = 1.0 if (sentiment_score > 0.3 and has_strong_assertion) else 0.5
            
            # Novelty: Check cache
            sent_hash = hashlib.md5(sent.encode()).hexdigest()
            novelty = 0.5 if sent_hash in cache else 0.8
            
            # Priority calculation
            priority = (0.4 * specificity) + (0.4 * controversy) + (0.2 * novelty)
            if priority > 0.7: 
                prioritized.append((sent, priority))
        except Exception as e:
            logger.error(f"Prioritization failed for '{sent}': {e}")
    return prioritized

# Step 4: API Call Management
class FactCheckManager:
    def __init__(self, api_key, max_calls_per_hour=1000, cooldown=5):
        self.api_key = api_key
        self.queue = deque()
        self.cache = set()
        self.lock = threading.Lock()
        self.call_count = 0
        self.max_calls = max_calls_per_hour
        self.cooldown = cooldown
        self.last_call_time = 0
        self.priority_threshold = 0.7 

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
        while True:
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
                        self.queue.popleft()  # Discard low-priority
                
                if batch:
                    self.call_count += 1
                    results = real_fact_check_api(batch, self.api_key)
                    self.last_call_time = current_time
                    for claim, (verdict, evidence) in zip(batch, results):
                        logger.info(f"Claim: '{claim}' → Verdict: {verdict}, Evidence: {evidence}")
                    time.sleep(1)  # Simulate latency

def fact_check_pipeline(text_stream, manager):
    for text_chunk in text_stream:
        logger.info(f"Processing chunk: '{text_chunk}'")
        sentences = segment_sentences(text_chunk)
        factual_sentences = classify_factual(sentences)
        prioritized_claims = prioritize_claims(factual_sentences, manager.cache)
        manager.add_to_queue(prioritized_claims)

input_stream = [
    "The Earth is flat and I love flat maps because the moon is made of cheese.",
    "Climate change is a hoax. I think summers are nicer now.",
    "Is the sky blue? One plus one equals two. Umm, yeah, so like...",
    "Cabrilla Lattuan has called well had been on the run for four years"
]

manager = FactCheckManager(api_key="dummy_key")
thread = threading.Thread(target=manager.process_queue, daemon=True)
thread.start()

fact_check_pipeline(input_stream, manager)
time.sleep(5) 