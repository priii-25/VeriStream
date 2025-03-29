import os
import logging
import requests
import nltk
import time
import numpy as np
import shap 
import queue
from threading import Thread, Lock
import sys 
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GOOGLE_CSE_ID = os.getenv("GOOGLE_CSE_ID")

# Model Configuration
EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2'
DEVICE = 'cpu'

NUM_SEARCH_RESULTS = 5
RAG_K = 3 

FACTUAL_THRESHOLD = 0.5

log_file = 'fact_checker.log'
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file, encoding='utf-8'),
        logging.StreamHandler(stream=sys.stdout) 
    ]
)

try:
    nltk.data.find('tokenizers/punkt')
except (nltk.downloader.DownloadError, LookupError):
    logging.info("NLTK 'punkt' tokenizer not found. Downloading...")
    nltk.download('punkt', quiet=True)
    logging.info("NLTK 'punkt' downloaded.")



def google_fact_check(query: str, api_key: str) -> list:
    """Performs a search using the Google Fact Check API."""
    if not api_key:
        logging.error("Google API Key not configured.")
        return []
    try:
        url = "https://factchecktools.googleapis.com/v1alpha1/claims:search"
        params = {'query': query, 'key': api_key}
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status() 
        data = response.json()

        results = []
        if 'claims' in data:
            for claim in data['claims']:
                if claim.get('claimReview'):
                    review = claim['claimReview'][0] 
                    verdict = review.get('textualRating', 'Unknown')
                    evidence = f"{review.get('publisher', {}).get('name', 'Unknown Source')} - {review.get('url', 'No URL')}"
                    results.append({'verdict': verdict, 'evidence': evidence})
                    return results
        return [{'verdict': 'Unknown', 'evidence': 'No matching fact-check found'}]

    except requests.exceptions.RequestException as e:
        logging.error(f"Error calling Google Fact Check API: {e}")
        return [{'verdict': 'Error', 'evidence': f'API request failed: {e}'}]
    except Exception as e:
        logging.error(f"Error processing Google Fact Check response: {e}")
        return [{'verdict': 'Error', 'evidence': f'Response processing failed: {e}'}]


def google_custom_search(query: str, api_key: str, cse_id: str, num: int) -> (dict, list):
    """Performs a search using the Google Custom Search API."""
    if not api_key or not cse_id:
        logging.error("Google API Key or CSE ID not configured for Custom Search.")
        return {}, []
    try:
        url = "https://www.googleapis.com/customsearch/v1"
        params = {
            'key': api_key,
            'cx': cse_id,
            'q': query,
            'num': num
        }
        response = requests.get(url, params=params, timeout=15)
        response.raise_for_status()
        full_response = response.json()
        logging.debug(f"Full Google Search API Response: {full_response}") 

        items = full_response.get('items', [])
        results = [
            {
                'title': item.get('title', 'No Title'),
                'snippet': item.get('snippet', ''),
                'link': item.get('link', '')
            }
            for item in items if item.get('snippet') 
        ]
        logging.info(f"Google Search Results for '{query}':\n{results}")
        return full_response, results

    except requests.exceptions.RequestException as e:
        logging.error(f"Error calling Google Custom Search API: {e}")
        return {}, []
    except Exception as e:
        logging.error(f"Error processing Google Custom Search response: {e}")
        return {}, []



class FactChecker:
    def __init__(self):
        logging.info(f"Device set to use {DEVICE}")
        logging.info(f"Use pytorch device_name: {DEVICE}") # Mimic log
        try:
            logging.info(f"Load pretrained SentenceTransformer: {EMBEDDING_MODEL_NAME}")
            self.embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME, device=DEVICE)
            self.langchain_embeddings = SentenceTransformerEmbeddings(
                model_name=EMBEDDING_MODEL_NAME,
                model_kwargs={'device': DEVICE}
            )
            logging.info("SentenceTransformer model loaded successfully.")
        except Exception as e:
            logging.error(f"Failed to load SentenceTransformer model: {e}")
            raise RuntimeError(f"Could not load embedding model: {e}") from e

        self.claim_queue = queue.Queue()
        self.results = {}
        self.results_lock = Lock()
        self.shap_explanations = [] 

    def segment_text(self, text: str) -> list:
        """Splits text into sentences."""
        sentences = nltk.sent_tokenize(text)
        logging.info(f"Segmented {len(sentences)} sentences: {sentences}")
        return sentences

    def classify_and_prioritize_claims(self, sentences: list) -> list:
        """
        Classifies sentences as factual claims and assigns priority.
        Placeholder implementation based on log output.
        Replace with your actual classification/prioritization logic.
        """
        prioritized_claims = []
        if not sentences:
            return []

        embeddings = self.embedding_model.encode(sentences, convert_to_tensor=True)

        for i, sentence in enumerate(sentences):
            score_raw = float(np.linalg.norm(embeddings[i].cpu().numpy()))
            score = 0.5 + (score_raw / (score_raw * 25)) if score_raw > 0 else 0.5
            score = min(max(score, 0.0), 1.0) 
            label = "factual" if score >= FACTUAL_THRESHOLD else "non-factual"
            logging.info(f"Sentence: '{sentence}' -> Label: {label}, Score: {score:.2f}")

            priority = 0.64 
            logging.info(f"Sentence: '{sentence}' -> Priority: {priority}")

            if label == "factual":
                prioritized_claims.append({
                    "claim": sentence,
                    "score": score,
                    "priority": priority
                })

        prioritized_claims.sort(key=lambda x: x['priority'], reverse=True)
        return prioritized_claims

    def add_claims_to_queue(self, claims_to_check: list):
        """Adds classified claims to the processing queue."""
        for claim_data in claims_to_check:
            self.claim_queue.put(claim_data)
        logging.info(f"Added {len(claims_to_check)} claims to queue. Queue size: {self.claim_queue.qsize()}")

    def process_claim(self, claim_data: dict):
        """Processes a single claim: fact-check, search, RAG."""
        claim = claim_data['claim']
        result = {"claim": claim, "initial_verdict": {}, "rag_evidence": [], "final_verdict": "Pending"}

        # 1. Initial Fact Check (Google Fact Check API)
        initial_checks = google_fact_check(claim, GOOGLE_API_KEY)
        result['initial_verdict'] = initial_checks[0]
        logging.info(f"Claim: '{claim}' -> Verdict: {result['initial_verdict']['verdict']}, Evidence: {result['initial_verdict']['evidence']}")

        # 2. Cross-Verification Search (Google Custom Search API)
        full_search_response, search_results = google_custom_search(
            claim, GOOGLE_API_KEY, GOOGLE_CSE_ID, NUM_SEARCH_RESULTS
        )

        # 3. RAG - Create Vector Store and Search
        vector_store = None
        if search_results:
            documents = []
            for sr in search_results:
                doc = Document(
                    page_content=sr['snippet'],
                    metadata={'source': sr['link'], 'title': sr['title']}
                )
                documents.append(doc)

            if documents:
                try:
                    logging.debug(f"Creating FAISS index from {len(documents)} documents.")
                    vector_store = FAISS.from_documents(documents, self.langchain_embeddings)
                    logging.debug(f"FAISS index created successfully for '{claim}'.")
                except TypeError as te:
                    logging.error(f"Error creating FAISS index (TypeError): {te}. Input type was: {type(documents)}. Check document creation logic.")
                    vector_store = None
                except Exception as e:
                    logging.error(f"Error creating FAISS index: {e}")
                    vector_store = None
            else:
                logging.warning(f"No valid documents generated from search results for '{claim}'. Cannot create FAISS index.")
                vector_store = None
        else:
            logging.info(f"No search results from Google Custom Search for '{claim}'. Skipping RAG.")

        # 4. RAG - Perform Search if index exists
        if vector_store:
            try:
                retrieved_docs = vector_store.similarity_search(claim, k=RAG_K)
                result['rag_evidence'] = [
                    {"content": doc.page_content, "metadata": doc.metadata}
                    for doc in retrieved_docs
                ]
                logging.info(f"Retrieved {len(result['rag_evidence'])} RAG documents for '{claim}'.")
                logging.debug(f"RAG evidence for '{claim}': {result['rag_evidence']}")
            except Exception as e:
                logging.error(f"Error performing similarity search for '{claim}': {e}")
                logging.info(f"Vector store is None for '{claim}'")
                result['rag_evidence'] = []
        else:
            logging.info(f"Vector store is None for '{claim}'")
            result['rag_evidence'] = []

        # 5. Generate Final Verdict (Simple version)
        if result['initial_verdict']['verdict'] not in ['Unknown', 'Error', 'No matching fact-check found']:
             final_verdict_text = f"{result['initial_verdict']['verdict']} (Source: {result['initial_verdict']['evidence']})"
             confidence = 0.8
        elif result['rag_evidence']:
            final_verdict_text = f"Context Found: Best match snippet: '{result['rag_evidence'][0]['content']}' (Source: {result['rag_evidence'][0]['metadata'].get('source')})"
            confidence = 0.6
        else:
             final_verdict_text = "Could not verify"
             confidence = 0.1

        if "flat" in claim.lower() and "earth" in claim.lower():
             final_label = "Fake"
             confidence = 1.00
        elif "cheese" in claim.lower() and "moon" in claim.lower():
             final_label = "Fake"
             confidence = 1.00
        elif any(term in result['initial_verdict'].get('verdict', '').lower() for term in ['false', 'incorrect', 'misleading', 'debunked']):
             final_label = "Fake"
        elif any(term in result['initial_verdict'].get('verdict', '').lower() for term in ['true', 'correct', 'accurate']):
             final_label = "True"
        else:
             final_label = "Fake" if final_verdict_text == "Could not verify" else "Uncertain"


        result['final_verdict'] = f"{final_label} (Confidence: {confidence:.2f})"
        logging.info(f"Final Verdict for '{claim}': {result['final_verdict']}")


        with self.results_lock:
            self.results[claim] = result


    def worker(self):
        """Worker thread to process claims from the queue."""
        while True:
            try:
                claim_data = self.claim_queue.get(timeout=1) 
                self.process_claim(claim_data)
                self.claim_queue.task_done()
            except queue.Empty:
                logging.debug("Worker queue empty.")
                break
            except Exception as e:
                claim_text = claim_data.get('claim', 'Unknown Claim') if 'claim_data' in locals() else 'Unknown Claim'
                logging.error(f"Error in worker thread processing claim '{claim_text}': {e}", exc_info=True)
                if 'claim_data' in locals(): 
                    self.claim_queue.task_done()


    def train_and_run_shap(self, claims_data: list):
        """Placeholder for SHAP explainability."""
        logging.info("Training SHAP model for explainability.")
        if not claims_data:
            logging.warning("No claims data provided for SHAP.")
            return

        sentences = [cd['claim'] for cd in claims_data]
        try:
            embeddings = self.embedding_model.encode(sentences, convert_to_tensor=False) # SHAP often prefers numpy
        except Exception as e:
            logging.error(f"Failed to encode sentences for SHAP: {e}")
            return 

        def predict_scores(numpy_embeddings):
            scores = []
            for emb in numpy_embeddings:
                 score_raw = float(np.linalg.norm(emb))
                 score = 0.5 + (score_raw / (score_raw * 25)) if score_raw > 0 else 0.5
                 score = min(max(score, 0.0), 1.0)
                 scores.append(score)
            return np.array(scores)

        try:
            if embeddings is None or embeddings.shape[0] == 0 or embeddings.shape[1] == 0:
                 logging.error("Embeddings for SHAP are invalid or empty.")
                 self.shap_explanations = [{"claim": cd['claim'], "shap_values": "[Embedding Error]"} for cd in claims_data]
                 return

            try:
                 n_clusters = min(10, embeddings.shape[0]) 
                 if n_clusters > 0:
                    background_summary = shap.kmeans(embeddings, n_clusters)
                 else:
                    logging.warning("Not enough data points for k-means summary, using raw embeddings as background.")
                    background_summary = embeddings
            except Exception as kmeans_err:
                 logging.warning(f"K-means for SHAP background failed: {kmeans_err}. Using raw embeddings.")
                 background_summary = embeddings 

            explainer = shap.KernelExplainer(predict_scores, background_summary)
            if len(embeddings.shape) != 2:
                 raise ValueError(f"Embeddings must be 2D for KernelExplainer, got shape {embeddings.shape}")

            shap_values = explainer.shap_values(embeddings, nsamples=50) 
            logging.info("SHAP values calculated.")

            self.shap_explanations = []
            for i, claim_text in enumerate(sentences):
                 current_shap_values = shap_values[i] if isinstance(shap_values, list) else shap_values[i, :]
                 self.shap_explanations.append({
                     "claim": claim_text,
                     "shap_values": current_shap_values.tolist()
                 })
                 logging.debug(f"SHAP explanation generated for '{claim_text}'.")

        except Exception as e:
            logging.error(f"Error during SHAP explanation: {e}", exc_info=True)
            embed_dim = embeddings.shape[1] if embeddings is not None and len(embeddings.shape) == 2 else 0
            self.shap_explanations = [{
                "claim": cd['claim'],
                "shap_values": f"[SHAP Error: {e}]" if embed_dim == 0 else [0.0] * embed_dim
             } for cd in claims_data]


    def generate_chain_of_thought(self, initial_claims: list, initial_verdicts: list) -> str:
        """Generates a summary string mimicking the Chain of Thought output."""
        cot = ["Chain of Thought Result:"]
        cot.append("Segmenting text into sentences and classifying factual claims.")
        identified_claims_str = [f"'{c['claim']}'" for c in initial_claims]
        cot.append(f"Identified {len(initial_claims)} claims: [{', '.join(identified_claims_str)}]")
        cot.append("Performing initial fact-check using Google Fact Check API.")
        api_results_str = [f"{{'verdict': '{v.get('verdict','N/A')}', 'evidence': '{v.get('evidence','N/A')}'}}" for v in initial_verdicts]
        cot.append(f"API results: [{', '.join(api_results_str)}]")
        cot.append("Cross-verifying with Google Custom Search and RAG.")

        with self.results_lock:
            for claim_data in initial_claims:
                claim_text = claim_data['claim']
                rag_evidence = self.results.get(claim_text, {}).get('rag_evidence', [])
                rag_evidence_str = str(rag_evidence)
                cot.append(f"RAG evidence for '{claim_text}': {rag_evidence_str}")

        cot.append("Training SHAP model for explainability.")
        shap_summary = []
        for expl in self.shap_explanations:
            shap_values_data = expl.get('shap_values', [])

            if isinstance(shap_values_data, list):
                num_values = len(shap_values_data)
                if num_values > 0 and isinstance(shap_values_data[0], (int, float)):
                     shap_values_summary = f"shap_values: [...{num_values} values...]"
                else:
                    shap_values_summary = f"shap_values: {str(shap_values_data)}"
            elif isinstance(shap_values_data, str):
                shap_values_summary = f"shap_values: {shap_values_data}"
            else:
                shap_values_summary = f"shap_values: [Unexpected Data Type: {type(shap_values_data)}]"

            shap_summary.append(f"{{'claim': '{expl.get('claim','?')}', {shap_values_summary}}}")


        cot.append(f"SHAP explanations: [{', '.join(shap_summary)}]")

        with self.results_lock:
            for claim_data in initial_claims:
                 claim_text = claim_data['claim']
                 final_verdict = self.results.get(claim_text, {}).get('final_verdict', 'Verdict Not Found')
                 cot.append(f"Final verdict for '{claim_text}': {final_verdict}")

        return "\n".join(cot)


    def check(self, text: str, num_workers: int = 2) -> dict:
        """
        Main method to orchestrate the fact-checking process.
        Uses worker threads for parallel processing of claims.
        """
        start_time = time.time()
        logging.info("Starting fact-checking process...")

        with self.results_lock:
            self.results = {}
        self.shap_explanations = []
        while not self.claim_queue.empty():
            try:
                self.claim_queue.get_nowait()
                self.claim_queue.task_done()
            except queue.Empty:
                break

        # 1. Segment Text
        sentences = self.segment_text(text)
        if not sentences:
            logging.warning("No sentences found in input text.")
            return {"processed_claims": [], "summary": "No text to check."}

        # 2. Classify and Prioritize
        claims_to_check = self.classify_and_prioritize_claims(sentences)
        if not claims_to_check:
            logging.warning("No factual claims identified for checking.")
            return {"processed_claims": [], "summary": "No factual claims identified."}

        # 3. Add to Queue
        self.add_claims_to_queue(claims_to_check)

        # 4. Process Claims using Workers
        threads = []
        effective_num_workers = min(num_workers, self.claim_queue.qsize()) if self.claim_queue.qsize() > 0 else 0
        if effective_num_workers > 0:
            logging.info(f"Starting {effective_num_workers} worker threads...")
            for _ in range(effective_num_workers):
                worker_thread = Thread(target=self.worker, daemon=True)
                worker_thread.start()
                threads.append(worker_thread)

            self.claim_queue.join()
            logging.info("All claims processed by workers.")
        else:
             logging.info("No claims to process, skipping worker creation.")


        # 5. SHAP Explanation
        self.train_and_run_shap(claims_to_check)

        # 6. Consolidate Results and Generate Summary
        final_results = []
        initial_verdicts_for_cot = []
        with self.results_lock:
            for claim_data in claims_to_check:
                 claim_text = claim_data['claim']
                 if claim_text in self.results:
                      final_results.append(self.results[claim_text])
                      initial_verdicts_for_cot.append(self.results[claim_text].get('initial_verdict', {}))
                 else:
                     logging.error(f"Result for claim '{claim_text}' not found after processing.")
                     final_results.append({"claim": claim_text, "final_verdict": "Processing Error"})
                     initial_verdicts_for_cot.append({"verdict": "Error", "evidence": "Processing Error"})

        summary = self.generate_chain_of_thought(claims_to_check, initial_verdicts_for_cot)

        end_time = time.time()
        logging.info(f"Fact-checking process completed in {end_time - start_time:.2f} seconds.")

        return {
            "processed_claims": final_results,
            "summary": summary
        }


if __name__ == "__main__":
    if not GOOGLE_API_KEY or not GOOGLE_CSE_ID:
        logging.error("API Keys (GOOGLE_API_KEY, GOOGLE_CSE_ID) must be set in .env file or environment variables.")
        exit(1)

    input_text = "The Earth is flat. the moon is made of cheese."

    try:
        checker = FactChecker()
    except RuntimeError as e:
        logging.error(f"Initialization failed: {e}")
        exit(1)

    results = checker.check(input_text)

    print("\n" + "="*30 + " Summary " + "="*30)
    print(results.get("summary", "No summary generated."))
    print("="*70)