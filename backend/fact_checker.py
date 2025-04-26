# backend/fact_checker.py
# ... (Imports and other unchanged code remain the same) ...
import os
import logging
import requests
import nltk
import time
import numpy as np
import string
import json
# --- LLM/SHAP/NLP Imports ---
try:
    import shap
except ImportError: shap = None
try:
    from groq import Groq, APIConnectionError, RateLimitError, APIStatusError
except ImportError:
    Groq = None
    class APIConnectionError(Exception): pass
    class RateLimitError(Exception): pass
    class APIStatusError(Exception): pass
try:
    import spacy
    try:
        NLP = spacy.load("en_core_web_sm")
        logging.info("spaCy model 'en_core_web_sm' loaded.")
    except OSError:
        logging.error("spaCy model 'en_core_web_sm' not found. Download: python -m spacy download en_core_web_sm")
        NLP = None
except ImportError:
    spacy = None
    NLP = None
# --- End Imports ---
import queue
from threading import Thread, Lock, current_thread
import sys
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
# Import Neo4j explicitly with auth and exceptions
from neo4j import GraphDatabase, basic_auth, exceptions as neo4j_exceptions

# --- Configuration ---
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'; os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
try: import tensorflow as tf; tf.get_logger().setLevel('ERROR')
except ImportError: pass
import warnings
from langchain_core._api.deprecation import LangChainDeprecationWarning
warnings.filterwarnings("ignore", message="The class `HuggingFaceEmbeddings` was deprecated*", category=LangChainDeprecationWarning)
# warnings.filterwarnings("ignore", message="The query used a deprecated function: `id`", category=UserWarning) # Optional: Suppress Neo4j warnings

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GOOGLE_CSE_ID = os.getenv("GOOGLE_CSE_ID")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# --- Neo4j Configuration ---
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "priyanshi") # Ensure this is correct
NEO4J_DATABASE = "veristream"
KG_CONFIDENCE_THRESHOLD = 0.85 # Minimum confidence required to trust KG verdict

# Clients
groq_client = None
if Groq and GROQ_API_KEY:
    try: groq_client = Groq(api_key=GROQ_API_KEY); logging.info("Groq client initialized.")
    except Exception as e: logging.error(f"Failed Groq client init: {e}")
elif not Groq: logging.warning("Groq library not installed. LLM disabled.")
elif not GROQ_API_KEY: logging.warning("GROQ_API_KEY not found. LLM disabled.")

# Models & Parameters
EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2'
DEVICE = 'cpu' # Or 'cuda' if GPU is available and configured
LLM_PROVIDER = "Groq"
LLM_MODEL_NAME = "llama3-8b-8192"
NUM_SEARCH_RESULTS = 5 # Increase slightly?
RAG_K = 5            # Increase slightly?
# Keywords for filtering non-checkable claims
OPINION_PHRASES = ["i think", "i believe", "in my opinion", "seems like", "feels like", "should be", "must be"]
SUBJECTIVE_ADJECTIVES = ["beautiful", "ugly", "amazing", "terrible", "wonderful", "awful", "best", "worst", "nice", "bad", "good", "great"] # Added great
SELF_REFERENCE_WORDS = ["this sentence", "this claim", "this statement", "i say", "i state", "this phrase"]
KG_RELEVANT_NER_LABELS = {"PERSON", "ORG", "GPE", "LOC", "DATE", "TIME", "MONEY", "QUANTITY", "PERCENT", "CARDINAL", "ORDINAL", "PRODUCT", "EVENT", "WORK_OF_ART", "LAW", "NORP", "FAC"}

# Logging
log_file = 'fact_checker.log'
with open(log_file, 'w', encoding='utf-8') as f: pass
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(threadName)s] - %(message)s', handlers=[logging.FileHandler(log_file, encoding='utf-8'), logging.StreamHandler(stream=sys.stdout)])

# NLTK
try: nltk.data.find('tokenizers/punkt')
except: logging.info("Downloading NLTK 'punkt'..."); nltk.download('punkt', quiet=True); logging.info("'punkt' downloaded.")

try: nltk.data.find('tokenizers/punkt_tab')
except LookupError: # More specific exception
    logging.info("Downloading NLTK 'punkt_tab' resource...")
    nltk.download('punkt_tab', quiet=True)
    logging.info("NLTK 'punkt_tab' downloaded.")

# --- API Functions (unchanged) ---
def google_fact_check(query: str, api_key: str) -> list:
    # Uses preprocessed query
    if not api_key: logging.error("GFactCheck Key missing."); return [{'verdict': 'Error', 'evidence': 'API Key missing'}]
    logging.debug(f"Querying GFactCheck API with: '{query}'")
    try:
        url="https://factchecktools.googleapis.com/v1alpha1/claims:search"; params={'query':query,'key':api_key,'languageCode':'en'}
        response = requests.get(url, params=params, timeout=10); response.raise_for_status(); data = response.json()
        if 'claims' in data and data['claims']:
            claim = data['claims'][0]
            if claim.get('claimReview'):
                rev=claim['claimReview'][0]; verdict=rev.get('textualRating','Unknown'); ev=f"{rev.get('publisher',{}).get('name','?')} - {rev.get('url','No URL')}"; return [{'verdict': verdict, 'evidence': ev}]
        return [{'verdict': 'Unknown', 'evidence': 'No matching fact-check found'}]
    except requests.exceptions.Timeout: logging.error(f"Timeout GFactCheck: '{query}'"); return [{'verdict': 'Error', 'evidence': 'API timeout'}]
    except requests.exceptions.RequestException as e: logging.error(f"Error GFactCheck API: {e}"); return [{'verdict': 'Error', 'evidence': f'API failed: {e}'}]
    except Exception as e: logging.error(f"Error processing GFactCheck: {e}"); return [{'verdict': 'Error', 'evidence': f'Response processing failed: {e}'}]

def google_custom_search(query: str, api_key: str, cse_id: str, num: int) -> (dict, list):
    # Uses preprocessed query
    if not api_key or not cse_id: logging.error("GCustomSearch keys missing."); return {}, []
    logging.debug(f"Querying GCustomSearch API with: '{query}'")
    try:
        url="https://www.googleapis.com/customsearch/v1"; params={'key':api_key,'cx':cse_id,'q':query,'num':num}
        response = requests.get(url, params=params, timeout=15); response.raise_for_status(); full_response = response.json()
        items = full_response.get('items', []); results = [{'title':item.get('title','?'),'snippet':item.get('snippet','').replace('\n',' '),'link':item.get('link','')} for item in items if item.get('snippet')]
        logging.info(f"GCustomSearch got {len(results)} results for query '{query}'."); return full_response, results
    except requests.exceptions.Timeout: logging.error(f"Timeout GCustomSearch: '{query}'"); return {}, []
    except requests.exceptions.RequestException as e: logging.error(f"Error GCustomSearch API: {e}"); return {}, [] # Graceful handle 429
    except Exception as e: logging.error(f"Error processing GCustomSearch: {e}"); return {}, []


# --- *** REVISED LLM Final Verdict Function *** ---
def get_llm_final_verdict(
    claim: str,
    initial_check_verdict: str,
    initial_check_evidence: str,
    rag_evidence: list,
    rag_status_msg: str
) -> dict:
    """
    Uses Groq LLM to determine a final verdict, confidence, and justification
    by synthesizing the initial check results and the RAG evidence.
    """
    if not groq_client:
        return {"final_label": "LLM Error", "confidence": 0.1, "explanation": "Groq client not initialized."}

    # --- Format Inputs for Prompt ---
    # Initial Check Info
    initial_check_info = "Initial Check Data:\n"
    if initial_check_verdict and initial_check_verdict != "Error" and initial_check_verdict != "N/A":
        initial_check_info += f"- Verdict: {initial_check_verdict}\n"
        initial_check_info += f"- Evidence: {initial_check_evidence}\n"
    else:
        initial_check_info += "- Verdict: Not Available or Error\n"

    # RAG Info
    rag_info = f"Retrieved Web Snippets (RAG Status: {rag_status_msg}):\n"
    if rag_evidence:
        for i, doc in enumerate(rag_evidence):
            snippet_text = doc.get('content', 'N/A').replace('"', "'") # Avoid breaking JSON
            source_text = doc.get('metadata', {}).get('source', 'N/A')
            rag_info += f"{i+1}. Snippet: \"{snippet_text}\"\n   Source: {source_text}\n"
    else:
        rag_info += "None Provided or Retrieval Failed.\n"
    # --- End Input Formatting ---

    # --- Revised Prompt for Synthesis ---
    prompt = f"""You are an expert fact-checker synthesizing evidence. Your goal is to determine the most accurate final verdict (True/False) and confidence level for the 'Claim' based on ALL the provided information: the 'Initial Check Data' (from a preliminary API check) and the 'Retrieved Web Snippets' (from RAG search).

Claim: "{claim}"

{initial_check_info}
{rag_info}

**Synthesis Task:**
1.  **Analyze all Evidence:** Critically evaluate both the Initial Check Data (if available and reliable) and the RAG snippets. Consider:
    *   Does the Initial Check provide a clear verdict? Is its source credible?
    *   Do the RAG snippets support, contradict, or are they irrelevant to the claim? How consistent and strong is the RAG evidence?
    *   How do the Initial Check and RAG evidence align or conflict?
2.  **Determine Final Verdict (True/False):** Based on your synthesis of *all* evidence:
    *   "True": If the combined evidence strongly and consistently supports the claim. Priority should be given to strong RAG evidence if it contradicts a weak or 'Unknown' initial check.
    *   "False": If the combined evidence strongly contradicts the claim OR if the overall evidence (considering both sources) is insufficient, ambiguous, or conflicting, preventing verification. **If neither source provides clear verification, the verdict is False.**
3.  **Assign Confidence Score (0.0 - 1.0):** Reflect your certainty in the final verdict based on the *strength, consistency, and agreement* of the evidence:
    *   *High Confidence (> 0.85):* Strong agreement between a reliable Initial Check and supporting/contradicting RAG, OR very strong RAG evidence that clearly outweighs an 'Unknown' or weak Initial Check.
    *   *Medium Confidence (0.6 - 0.85):* One source is strong while the other is weak/ambiguous/missing, OR moderate agreement/disagreement between sources. RAG findings partially supporting/contradicting.
    *   *Low Confidence (<= 0.6):* Both sources are weak, conflicting, irrelevant, or RAG failed significantly, leading to insufficient evidence overall. **A 'False' verdict due to overall lack of evidence warrants low confidence.**
4.  **Provide Justification:** Explain your reasoning in 1-2 sentences, explicitly mentioning how you weighed the Initial Check vs. RAG evidence to arrive at the final verdict and confidence. (e.g., "RAG snippets consistently contradicted the claim, overriding the 'Unknown' initial check.", "Initial check indicated 'False' and RAG failed to find relevant info, supporting 'False' with medium confidence due to lack of RAG confirmation.", "Initial check was 'Error' and RAG snippets were irrelevant, leading to 'False' verdict due to lack of evidence.")

**Output Format:** Respond with *only* a valid JSON object containing exactly three keys: "verdict" (string: "True" or "False"), "confidence" (float: 0.0-1.0), "justification" (string).

Example (RAG overrides Initial):
{{
  "verdict": "True",
  "confidence": 0.9,
  "justification": "While the initial check was 'Unknown', multiple relevant RAG snippets (1, 3) strongly supported the claim's core assertion."
}}

Example (Initial + Weak RAG):
{{
  "verdict": "False",
  "confidence": 0.7,
  "justification": "The initial check rated the claim 'False'. RAG provided only tangential snippets (1, 2), failing to contradict the initial check, thus maintaining a 'False' verdict with medium confidence."
}}

Example (Both sources weak/irrelevant):
{{
  "verdict": "False",
  "confidence": 0.35,
  "justification": "The initial check was 'Unknown' and the RAG snippets were entirely irrelevant to the claim, resulting in a 'False' verdict due to insufficient evidence overall."
}}

Your JSON Response:
"""
    logging.debug(f"LLM Synthesized Verdict Prompt for '{claim}':\n{prompt}")
    try:
        response = groq_client.chat.completions.create(
            model=LLM_MODEL_NAME,
            messages=[
                {"role": "system", "content": "Output only valid JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2, # Slightly higher temp might help synthesis? Let's try 0.2
            max_tokens=300 # Allow more tokens for synthesis explanation
        )
        llm_output_text = response.choices[0].message.content
        logging.debug(f"Groq LLM Raw: {llm_output_text}")

        # Parse JSON robustly (same parsing logic as before)
        try:
            llm_output_text_clean = llm_output_text.strip()
            if llm_output_text_clean.startswith("```json"):
                llm_output_text_clean = llm_output_text_clean[7:]
                if llm_output_text_clean.endswith("```"):
                    llm_output_text_clean = llm_output_text_clean[:-3]
            elif llm_output_text_clean.startswith("```"):
                 llm_output_text_clean = llm_output_text_clean[3:]
                 if llm_output_text_clean.endswith("```"):
                    llm_output_text_clean = llm_output_text_clean[:-3]

            llm_res = json.loads(llm_output_text_clean.strip())
            v = llm_res.get("verdict")
            c_raw = llm_res.get("confidence")
            j = llm_res.get("justification")

            if not (isinstance(v, str) and v in ["True", "False"]): raise ValueError(f"Invalid 'verdict' value: {v}")
            if not (isinstance(j, str) and j.strip()): raise ValueError(f"Invalid or empty 'justification': {j}")
            try:
                c = float(c_raw)
                if not (0.0 <= c <= 1.0):
                     logging.warning(f"LLM confidence {c} out of range [0.0, 1.0] for claim '{claim[:50]}...'. Clamping.")
                     c = max(0.0, min(1.0, c))
            except (TypeError, ValueError) as conf_e: raise ValueError(f"Invalid 'confidence' value: {c_raw}. Error: {conf_e}")

            return {"final_label": v, "confidence": c, "explanation": j}

        except json.JSONDecodeError as e: logging.error(f"LLM JSON decode fail for '{claim[:50]}...': {e}. Resp: {llm_output_text}"); return {"final_label": "LLM Error", "confidence": 0.1, "explanation": f"LLM returned non-JSON: {llm_output_text}"}
        except (KeyError, ValueError, TypeError) as e: logging.error(f"LLM invalid content for '{claim[:50]}...': {e}. Resp: {llm_output_text}"); return {"final_label": "LLM Error", "confidence": 0.1, "explanation": f"Invalid LLM JSON content ({e}): {llm_output_text}"}
        except Exception as e: logging.error(f"LLM unexpected parse error for '{claim[:50]}...': {e}. Resp: {llm_output_text}"); return {"final_label": "LLM Error", "confidence": 0.1, "explanation": f"LLM response parsing error: {llm_output_text}"}

    # Handle API errors
    except APIConnectionError as e: logging.error(f"Groq ConnErr for '{claim[:50]}...': {e}"); return {"final_label": "LLM Error", "confidence": 0.1, "explanation": f"API Conn Err: {e}"}
    except RateLimitError as e: logging.error(f"Groq RateLimit for '{claim[:50]}...': {e}"); time.sleep(5); return {"final_label": "LLM Error", "confidence": 0.1, "explanation": f"API Rate Limit: {e}"}
    except APIStatusError as e: logging.error(f"Groq API Status {e.status_code} for '{claim[:50]}...': {e.response}"); return {"final_label": "LLM Error", "confidence": 0.1, "explanation": f"API Status {e.status_code}"}
    except Exception as e: logging.error(f"Unexpected Groq API err for '{claim[:50]}...': {e}", exc_info=True); return {"final_label": "LLM Error", "confidence": 0.1, "explanation": f"API Err: {e}"}
# --- End Revised LLM Function ---

# --- Helper Functions (is_claim_checkable, preprocess_claim_for_api unchanged) ---
def is_claim_checkable(sentence: str) -> bool:
    # No changes needed here
    if not sentence or not isinstance(sentence, str): return False
    text_lower = sentence.lower().strip()
    if text_lower.startswith('"') and text_lower.endswith('"'): text_lower = text_lower[1:-1]
    if text_lower.startswith("'") and text_lower.endswith("'"): text_lower = text_lower[1:-1]
    for phrase in OPINION_PHRASES:
        if text_lower.startswith(phrase+" "): logging.info(f"Filtered (Opinion): '{sentence}'"); return False
    if any(word in text_lower for word in SELF_REFERENCE_WORDS): logging.info(f"Filtered (Self-Ref): '{sentence}'"); return False
    if text_lower.endswith("?"): logging.info(f"Filtered (Question): '{sentence}'"); return False
    if NLP:
        try:
            doc = NLP(sentence)
            if any(t.lemma_.lower() in SUBJECTIVE_ADJECTIVES for t in doc if t.pos_=="ADJ"): logging.info(f"Filtered (Subjective): '{sentence}'"); return False
            has_verb = any(t.pos_ in ("VERB","AUX") for t in doc); has_subj = any(t.dep_ in ("nsubj","nsubjpass","csubj","csubjpass","expl") for t in doc)
            if not (has_verb or has_subj) and len(doc)<5: logging.info(f"Filtered (Structure): '{sentence}'"); return False
            if len(doc)>0 and doc[0].pos_=="VERB" and doc[0].tag_=="VB":
                 is_imperative = not any(t.dep_.startswith("nsubj") for t in doc)
                 if is_imperative: logging.info(f"Filtered (Imperative): '{sentence}'"); return False
        except Exception as nlp_e:
            logging.error(f"spaCy processing error during checkability for '{sentence}': {nlp_e}")
            return False
    elif len(text_lower.split()) < 3:
         logging.info(f"Filtered (Short/No NLP): '{sentence}'"); return False
    return True

def preprocess_claim_for_api(original_claim: str) -> str:
    # No changes needed here
    if not NLP or not original_claim: return original_claim
    try:
        doc = NLP(original_claim); simplified_parts = []; root = None; subj = None; obj_or_comp = None; negation = False
        for token in doc:
            if token.dep_ == "ROOT": root = token
            if token.dep_ == "neg": negation = True
        if not root:
             if len(doc) < 5: return original_claim
             else: logging.warning(f"No ROOT for '{original_claim}', using original."); return original_claim
        subjects = [token for token in doc if "subj" in token.dep_]
        if subjects: subj = subjects[0]
        for child in root.children:
            if "obj" in child.dep_ or "attr" in child.dep_ or "acomp" in child.dep_:
                 obj_or_comp = child
                 while True:
                      potential_heads = [c for c in obj_or_comp.children if c.pos_ in ("NOUN", "PROPN", "ADJ", "NUM")]
                      if potential_heads: obj_or_comp = potential_heads[0]; continue
                      else: break
                 break
        if subj: simplified_parts.append(subj.text)
        if negation: simplified_parts.append("not")
        main_verb = root
        if root.pos_ != 'VERB' and root.head.pos_ == 'VERB': main_verb = root.head
        elif root.pos_ != 'VERB':
             aux_verb = next((t for t in root.children if t.pos_ == 'AUX'), None)
             if aux_verb: main_verb = aux_verb
        simplified_parts.append(main_verb.lemma_)
        if obj_or_comp: simplified_parts.append(obj_or_comp.text)
        simplified_claim = " ".join(simplified_parts)
        if len(simplified_claim.split()) < 2 or not simplified_claim.strip():
            logging.warning(f"Short simplification ('{simplified_claim}') for '{original_claim}', using original.");
            return original_claim.strip()
        logging.info(f"Simplified '{original_claim}' -> '{simplified_claim}' for API.");
        return simplified_claim.strip()
    except Exception as e: logging.error(f"Error preprocessing '{original_claim}': {e}"); return original_claim

# --- FactChecker Class (Initialization, KG Check, Neo4j Storage, Preprocessing, Prioritization - unchanged) ---
class FactChecker:
    def __init__(self):
        logging.info(f"Device set to use {DEVICE}")
        try:
            logging.info(f"Load ST: {EMBEDDING_MODEL_NAME}"); self.embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME, device=DEVICE)
            self.langchain_embeddings = SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL_NAME, model_kwargs={'device': DEVICE}); logging.info("Models loaded.")
        except Exception as e: logging.error(f"Model load failed: {e}"); raise RuntimeError(f"Could not load model: {e}") from e
        if shap is None: logging.warning("SHAP not installed.")
        self.shap_available = shap is not None
        if NLP is None: logging.warning("spaCy model not loaded. Claim filtering/preprocessing basic.")
        self.nlp_available = NLP is not None
        self.claim_queue=queue.Queue(); self.results={}; self.results_lock=Lock(); self.shap_explanations=[]
        self.raw_fact_checks={}; self.raw_searches={}

        # Neo4j Driver Initialization
        self.neo4j_driver = None
        try:
            self.neo4j_driver = GraphDatabase.driver(NEO4J_URI, auth=basic_auth(NEO4J_USER, NEO4J_PASSWORD))
            with self.neo4j_driver.session(database=NEO4J_DATABASE) as session:
                session.run("RETURN 1")
            logging.info(f"Neo4j driver initialized for database '{NEO4J_DATABASE}'.")
        except neo4j_exceptions.AuthError as auth_err:
             logging.error(f"Neo4j Authentication Failed for user '{NEO4J_USER}'. Check credentials. Error: {auth_err}")
             self.neo4j_driver = None
        except neo4j_exceptions.ServiceUnavailable as conn_err:
             logging.error(f"Neo4j Service Unavailable at URI '{NEO4J_URI}'. Check if Neo4j is running. Error: {conn_err}")
             self.neo4j_driver = None
        except Exception as e:
            logging.error(f"Failed to initialize Neo4j driver for URI '{NEO4J_URI}', DB '{NEO4J_DATABASE}': {e}", exc_info=True)
            self.neo4j_driver = None

    def check_kg_for_claim(self, preprocessed_claim: str) -> dict | None:
        """Queries Neo4j for a matching preprocessed claim with a reliable verdict."""
        if not self.neo4j_driver:
            logging.warning("KG Check skipped: Neo4j driver not available.")
            return None
        if not preprocessed_claim:
             logging.warning("KG Check skipped: Empty preprocessed claim provided.")
             return None

        logging.debug(f"KG Check: Querying for preprocessed claim: '{preprocessed_claim}'")
        query = """
        MATCH (c:Claim {preprocessed_text: $prep_text})-[:HAS_VERDICT]->(v:Verdict)
        WHERE v.verdict_label IN ["True", "False"] AND v.confidence >= $min_confidence
        RETURN c.text AS original_claim,
               v.verdict_label AS verdict_label,
               v.confidence AS confidence,
               v.explanation AS explanation,
               c.timestamp AS timestamp
        ORDER BY v.confidence DESC, c.timestamp DESC
        LIMIT 1
        """
        params = {"prep_text": preprocessed_claim, "min_confidence": KG_CONFIDENCE_THRESHOLD}

        try:
            with self.neo4j_driver.session(database=NEO4J_DATABASE) as session:
                result = session.run(query, params).single()

            if result:
                logging.info(f"KG HIT: Found existing verdict for '{preprocessed_claim}' -> '{result['verdict_label']}' (Conf: {result['confidence']:.2f})")
                kg_result_dict = {
                    "original_claim": result["original_claim"],
                    "preprocessed_claim": preprocessed_claim,
                    "ner_entities": [],
                    "factual_score": None,
                    "initial_verdict_raw": "From KG",
                    "initial_evidence": "From KG",
                    "rag_status": "N/A (From KG)",
                    "top_rag_snippets": [],
                    "final_label": result["verdict_label"],
                    "confidence": result["confidence"],
                    "final_explanation": result["explanation"],
                    "source": "Knowledge Graph",
                    "kg_timestamp": result["timestamp"]
                }
                return kg_result_dict
            else:
                logging.info(f"KG MISS: No reliable verdict found for '{preprocessed_claim}'.")
                return None
        except neo4j_exceptions.ServiceUnavailable as e:
            logging.error(f"KG Check Failed: Neo4j connection error: {e}")
            return None
        except Exception as e:
            logging.error(f"KG Check Failed: Error querying Neo4j for '{preprocessed_claim}': {e}", exc_info=True)
            return None

    def store_in_neo4j(self, claim_data):
        # Uses elementId() - No changes needed
        if not self.neo4j_driver:
           logging.error("Neo4j driver not initialized. Cannot store data.")
           return
        if claim_data.get("source") == "Knowledge Graph":
             logging.debug(f"Skipping Neo4j store for claim already retrieved from KG: '{claim_data.get('original_claim', '?')[:50]}...'")
             return

        claim = claim_data.get('original_claim', 'Unknown Claim')
        preprocessed_claim = claim_data.get('preprocessed_claim', '')
        final_label = claim_data.get('final_label', 'Error')
        confidence = claim_data.get('confidence', 0.1)
        final_explanation = claim_data.get('final_explanation', 'N/A')

        try:
            confidence = float(confidence)
            if not (0.0 <= confidence <= 1.0):
                 logging.warning(f"Confidence {confidence} out of range for claim '{claim[:50]}...'. Clamping.")
                 confidence = max(0.0, min(1.0, confidence))
        except (ValueError, TypeError):
            logging.warning(f"Invalid confidence type '{type(confidence)}' for claim '{claim[:50]}...'. Using 0.1.")
            confidence = 0.1

        if not isinstance(final_label, str):
            logging.warning(f"Invalid final_label type '{type(final_label)}' for claim '{claim[:50]}...'. Converting.")
            final_label = str(final_label)

        entities = claim_data.get('ner_entities', [])
        initial_evidence = claim_data.get('initial_evidence', "")
        rag_status = claim_data.get('rag_status', "")
        initial_verdict_raw = claim_data.get('initial_verdict_raw', "")
        factual_score = claim_data.get('factual_score', 0.0)
        if factual_score is None: factual_score = 0.0

        top_rag_snippets = claim_data.get('top_rag_snippets', [])
        timestamp = time.time()

        with self.neo4j_driver.session(database=NEO4J_DATABASE) as session:
            tx = None
            try:
                tx = session.begin_transaction()

                claim_node_result = tx.run("""
                    MERGE (c:Claim {preprocessed_text: $preprocessed_text})
                    ON CREATE SET c.text = $text, c.timestamp = $timestamp, c.initial_verdict_raw = $initial_verdict_raw, c.rag_status = $rag_status, c.initial_evidence = $initial_evidence, c.factual_score = $factual_score
                    ON MATCH SET c.timestamp = $timestamp, c.initial_verdict_raw = $initial_verdict_raw, c.rag_status = $rag_status, c.initial_evidence = $initial_evidence, c.factual_score = $factual_score, c.text = CASE WHEN c.text <> $text THEN $text ELSE c.text END
                    RETURN elementId(c) as claim_id
                """, text=claim, preprocessed_text=preprocessed_claim, timestamp=timestamp, initial_verdict_raw=initial_verdict_raw, rag_status=rag_status, initial_evidence=initial_evidence, factual_score=factual_score).single()

                if not claim_node_result: raise Exception(f"Claim node MERGE failed")
                claim_node_id = claim_node_result['claim_id']

                tx.run("MATCH (c:Claim) WHERE elementId(c) = $claim_node_id MATCH (c)-[r:HAS_VERDICT]->(old_v:Verdict) DELETE r, old_v", claim_node_id=claim_node_id)

                verdict_node_result = tx.run("""
                    CREATE (v:Verdict {verdict_label: $verdict_label, confidence: $confidence, explanation: $explanation})
                    RETURN elementId(v) as verdict_id
                    """, verdict_label=final_label, confidence=confidence, explanation=final_explanation).single()
                if not verdict_node_result: raise Exception("Verdict node creation failed.")
                verdict_id = verdict_node_result['verdict_id']

                tx.run("MATCH (c), (v) WHERE elementId(c) = $claim_node_id AND elementId(v) = $verdict_id CREATE (c)-[:HAS_VERDICT]->(v)", claim_node_id=claim_node_id, verdict_id=verdict_id)

                tx.run("MATCH (c:Claim) WHERE elementId(c) = $claim_node_id MATCH (c)-[r:HAS_EVIDENCE]->(old_es:EvidenceSnippet) DELETE r, old_es", claim_node_id=claim_node_id)

                for i, snippet_formatted_string in enumerate(top_rag_snippets):
                    source = "?"; content = snippet_formatted_string
                    try: # Basic parsing attempt
                        if '\" (' in snippet_formatted_string:
                             parts = snippet_formatted_string.split('\" (', 1)
                             content_part = parts[0]
                             source = parts[1][:-1] if len(parts)>1 and parts[1].endswith(')') else "?"
                             content = content_part.split(': "', 1)[1] if ': "' in content_part else content_part
                        else: content = snippet_formatted_string
                    except Exception as e: content = snippet_formatted_string # Fallback
                    evidence_snippet_node_result = tx.run("""
                        CREATE (es:EvidenceSnippet {content: $content, source: $source, snippet_index: $snippet_index})
                        RETURN elementId(es) as snippet_id
                        """, content=content, source=source, snippet_index=i).single()
                    if evidence_snippet_node_result:
                        evidence_snippet_id = evidence_snippet_node_result['snippet_id']
                        tx.run("MATCH (c), (es) WHERE elementId(c) = $claim_node_id AND elementId(es) = $evidence_snippet_id CREATE (c)-[:HAS_EVIDENCE]->(es)", claim_node_id=claim_node_id, evidence_snippet_id=evidence_snippet_id)

                for entity in entities:
                    entity_text = entity.get('text'); entity_label = entity.get('label')
                    if not entity_text or not entity_label: continue
                    entity_node_result = tx.run("MERGE (e:Entity {text: $text, label: $label}) RETURN elementId(e) as entity_id", text=entity_text, label=entity_label).single()
                    if entity_node_result:
                        entity_id = entity_node_result['entity_id']
                        tx.run("MATCH (c), (e) WHERE elementId(c) = $claim_node_id AND elementId(e) = $entity_id MERGE (c)-[:MENTIONS]->(e)", claim_node_id=claim_node_id, entity_id=entity_id)

                tx.commit()
                logging.info(f"Stored/Updated claim '{claim[:50]}...' in Neo4j.")

            except Exception as e:
                if tx:
                    logging.error(f"Transaction failed for claim '{claim[:50]}...'. Rolling back.")
                    try: tx.rollback()
                    except Exception as rb_e: logging.error(f"Error during rollback: {rb_e}")
                logging.error(f"Error storing/updating claim '{claim[:50]}...' in Neo4j: {e}", exc_info=True)

    def preprocess_and_filter(self, text: str) -> (list, list):
        # No changes needed here
        if not text or not isinstance(text, str): logging.warning("Preprocess: Input empty."); return [], []
        sentences = []; checkable_claims_data = []; non_checkable_claims = []
        try:
            text=text.replace('.This','. This').replace('?This','? This').replace('!This','! This');
            sentences=nltk.sent_tokenize(text);
            sentences=[s.strip() for s in sentences if len(s.strip()) > 1];
            logging.info(f"Segmented {len(sentences)} potential sentences.")
        except Exception as e: logging.error(f"Tokenization error: {e}"); return [], []

        for sentence in sentences:
            if is_claim_checkable(sentence):
                sentence_ner = [];
                if self.nlp_available:
                    try: doc=NLP(sentence); sentence_ner=[{"text":ent.text,"label":ent.label_} for ent in doc.ents if ent.label_ in KG_RELEVANT_NER_LABELS]; logging.debug(f"NER for '{sentence}': {sentence_ner}")
                    except Exception as e: logging.error(f"NER fail '{sentence}': {e}")
                preprocessed=preprocess_claim_for_api(sentence);
                # Ensure preprocessed text is not empty
                if not preprocessed:
                     logging.warning(f"Preprocessing resulted in empty string for '{sentence}'. Skipping.")
                     non_checkable_claims.append(sentence + " [Preprocessing Failed]")
                     continue
                checkable_claims_data.append({"original_claim":sentence,"preprocessed_claim":preprocessed,"ner_entities":sentence_ner})
            else:
                non_checkable_claims.append(sentence)
        logging.info(f"Found {len(checkable_claims_data)} checkable claims.");
        if non_checkable_claims: logging.info(f"Filtered {len(non_checkable_claims)} non-checkable sentences.")
        return checkable_claims_data, non_checkable_claims

    def classify_and_prioritize_claims(self, checkable_claims_data: list) -> list:
        # No changes needed here
        if not checkable_claims_data: logging.warning("Prioritize: No data."); return []
        try:
            # Filter out any entries that might lack 'original_claim' before encoding
            valid_claims_data = [cd for cd in checkable_claims_data if cd.get('original_claim')]
            if not valid_claims_data: return []
            orig=[cd["original_claim"] for cd in valid_claims_data];

            embeds=self.embedding_model.encode(orig,convert_to_tensor=True,show_progress_bar=False)
            for i,cd in enumerate(valid_claims_data):
                e=embeds[i].cpu().numpy();
                norm = np.linalg.norm(e)
                score = 0.5 + (norm / (norm * 25 + 1e-6)) if norm > 1e-6 else 0.5
                score = min(max(score, 0.0), 1.0)
                priority = score
                cd["factual_score"]=score;
                cd["priority"]=priority;
                logging.info(f"Prioritize: '{cd['original_claim'][:50]}...' -> Score:{score:.3f}, Prio:{priority:.3f}")
            valid_claims_data.sort(key=lambda x:x.get('priority', 0.0),reverse=True);
            logging.info(f"Prioritized {len(valid_claims_data)} claims.");
            return valid_claims_data
        except Exception as e: logging.error(f"Prioritization error: {e}",exc_info=True); return checkable_claims_data # Return original on error

    def add_claims_to_queue(self, claims_to_process: list):
        # No changes needed here
        if not claims_to_process: logging.warning("Queue: No claims."); return
        for cd_dict in claims_to_process: self.claim_queue.put(cd_dict)
        logging.info(f"Queued {len(claims_to_process)} claims for full processing. Queue Size: {self.claim_queue.qsize()}")

    # --- MODIFIED process_claim ---
    def process_claim(self, claim_data_dict: dict):
        """Pipeline for claims *not* found in KG: GFactCheck -> GSearch -> RAG -> Synthesizing LLM -> Neo4j Store."""
        original_claim = claim_data_dict.get('original_claim', '?')
        preprocessed_claim = claim_data_dict.get('preprocessed_claim', '?')
        ner_entities = claim_data_dict.get('ner_entities', [])
        start_time = time.time()
        logging.info(f"Full Process Start: \"{original_claim[:60]}...\" (API Query: \"{preprocessed_claim[:60]}...\")")

        result = {
            "original_claim": original_claim, "preprocessed_claim": preprocessed_claim, "ner_entities": ner_entities,
            "factual_score": claim_data_dict.get('factual_score', 0.0),
            "initial_verdict_raw": "N/A", "initial_evidence": "N/A", # Step 1 result
            "rag_status": "Not Attempted", "top_rag_snippets": [], # Step 2/3 results
            "final_label": "Pending", "confidence": 0.0, "final_explanation": "N/A", # Final results from synthesis
            "source": "Full Pipeline"
        }
        rag_evidence_for_llm = []

        # --- Step 1: Initial Fact Check (Google Fact Check API) ---
        initial_check_verdict = "N/A"
        initial_check_evidence = "N/A"
        try:
            initial_check_list = google_fact_check(preprocessed_claim, GOOGLE_API_KEY)
            self.raw_fact_checks[original_claim] = initial_check_list
            if initial_check_list:
                 initial_check = initial_check_list[0]
                 initial_check_verdict = initial_check.get('verdict', 'Error')
                 initial_check_evidence = initial_check.get('evidence', 'N/A')
                 result['initial_verdict_raw'] = initial_check_verdict
                 result['initial_evidence'] = initial_check_evidence
            else:
                 result['initial_verdict_raw'] = 'Error'; result['initial_evidence'] = 'API returned no data'
                 initial_check_verdict = 'Error' # Ensure these are set for LLM input
                 initial_check_evidence = 'API returned no data'
            logging.info(f"GFactCheck Result for '{original_claim[:50]}...': '{initial_check_verdict}'")
        except Exception as e:
            logging.error(f"GFactCheck fail for '{original_claim[:50]}...': {e}")
            result['initial_verdict_raw']="Error"; result['initial_evidence']=f"API fail:{e}"
            initial_check_verdict = 'Error'; initial_check_evidence = f"API fail:{e}"

        # --- Step 2: Google Custom Search (For RAG) ---
        search_results = []; full_search_resp = {}
        try:
            logging.debug(f"Attempting GCustomSearch '{preprocessed_claim[:60]}...'");
            # Maybe use original claim for search if preprocessing is poor?
            search_query = preprocessed_claim if len(preprocessed_claim.split()) > 1 else original_claim
            full_search_resp, search_results = google_custom_search(search_query, GOOGLE_API_KEY, GOOGLE_CSE_ID, NUM_SEARCH_RESULTS)
            self.raw_searches[original_claim] = {"query": search_query, "response": full_search_resp, "results": search_results}
            if not search_results and full_search_resp: result["rag_status"] = "Search OK, No Results"
            elif not search_results and not full_search_resp: result["rag_status"] = "Search Failed (API Error)"
            else: result["rag_status"] = "Search OK, Results Found"
            logging.info(f"GCustomSearch status for '{original_claim[:50]}...': {result['rag_status']}")
        except Exception as e: logging.error(f"GCustomSearch error for '{original_claim[:50]}...': {e}"); result["rag_status"]="Search Failed (Code Error)"; search_results=[]

        # --- Step 3: RAG (Vector Search on Search Results) ---
        if search_results:
            documents=[]; vector_store=None
            try:
                min_snippet_len = 20
                valid_search_results = [sr for sr in search_results if len(sr.get('snippet', '')) >= min_snippet_len]

                if not valid_search_results:
                     result["rag_status"] = "Search OK, No Usable Snippets"
                     logging.info(f"RAG: No snippets met minimum length for '{original_claim[:50]}...'")
                else:
                    # Use original claim words for relevance check
                    claim_words = {w.lower() for w in original_claim.translate(str.maketrans('', '', string.punctuation)).split() if len(w)>2}
                    # Relax relevance check slightly? Only require ONE claim word?
                    documents=[Document(page_content=sr['snippet'],metadata={'source':sr['link'],'title':sr['title']})
                               for sr in valid_search_results if any(w in sr['snippet'].lower() for w in claim_words)]

                    if documents:
                        try:
                            vector_store=FAISS.from_documents(documents,self.langchain_embeddings); logging.debug("FAISS index OK.")
                            retrieved_docs=vector_store.similarity_search(original_claim, k=RAG_K);
                            rag_evidence_for_llm=[{"content":doc.page_content,"metadata":doc.metadata} for doc in retrieved_docs]
                            result['top_rag_snippets']=[f"Snip {j+1}: \"{d['content'][:150].strip()}...\" ({d['metadata'].get('source','?')})" for j,d in enumerate(rag_evidence_for_llm)]
                            result["rag_status"] = f"RAG OK ({len(rag_evidence_for_llm)}/{len(documents)} snippets retrieved/indexed)"
                            logging.info(f"RAG status for '{original_claim[:50]}...': {result['rag_status']}")
                        except Exception as faiss_e:
                             logging.error(f"FAISS/Embedding error during RAG for '{original_claim[:50]}...': {faiss_e}")
                             result["rag_status"] = f"RAG Failed (FAISS Error)"
                             rag_evidence_for_llm = []
                             result['top_rag_snippets'] = []
                    else:
                        result["rag_status"] = "Search OK, No Relevant Docs Found for Index"
                        result['top_rag_snippets']=[]
                        rag_evidence_for_llm=[]
                        logging.info(f"RAG: No relevant documents found after filtering for '{original_claim[:50]}...'")

            except Exception as e: # Catch errors in the outer RAG try block
                logging.error(f"Outer RAG fail for '{original_claim[:50]}...': {e}",exc_info=True);
                result["rag_status"]=f"RAG Failed ({type(e).__name__})"
                rag_evidence_for_llm=[]
                result['top_rag_snippets']=[]
        else:
             result["rag_status"] = result.get("rag_status", "Search Returned No Results") if "Fail" in result.get("rag_status", "") else "Search Returned No Results"
             rag_evidence_for_llm = []
             result['top_rag_snippets'] = []

        # --- Step 4: LLM Final Verdict (Synthesizing Initial Check + RAG) ---
        logging.info(f"LLM generating final synthesized verdict for '{original_claim[:50]}...'...");
        # Pass the necessary info to the revised function
        llm_final_result = get_llm_final_verdict(
            claim=original_claim,
            initial_check_verdict=initial_check_verdict,
            initial_check_evidence=initial_check_evidence,
            rag_evidence=rag_evidence_for_llm,
            rag_status_msg=result['rag_status']
        )
        # Store the synthesized results
        result['final_label'] = llm_final_result['final_label']
        result['confidence'] = llm_final_result['confidence']
        result['final_explanation'] = llm_final_result['explanation']

        # --- Step 5: Store in Neo4j ---
        try:
            self.store_in_neo4j(result)
        except Exception as neo4j_e:
            logging.error(f"Neo4j storage failed for claim '{original_claim[:50]}...': {neo4j_e}", exc_info=True)

        processing_time = time.time() - start_time
        logging.info(f"Final Verdict (Synthesized) for '{original_claim[:50]}...': {result['final_label']} (Conf: {result['confidence']:.2f}). (Explain: {result['final_explanation'][:80]}...). (Time:{processing_time:.2f}s)")
        with self.results_lock: self.results[original_claim] = result # Store final synthesized result

    # --- Worker Function (unchanged) ---
    def worker(self):
        t_obj=current_thread(); t_name=t_obj.name; logging.info(f"W {t_name} start.")
        while True:
            cd_dict=None
            try:
                cd_dict=self.claim_queue.get(timeout=1)
                if cd_dict is None: break
                self.process_claim(cd_dict)
                self.claim_queue.task_done()
            except queue.Empty:
                logging.info(f"W {t_name} queue empty, finishing."); break
            except Exception as e:
                orig_claim=cd_dict.get('original_claim','?') if cd_dict else '?'; logging.error(f"W {t_name} error processing claim '{orig_claim[:50]}...': {e}",exc_info=True)
                if cd_dict:
                    with self.results_lock:
                        self.results[orig_claim]={
                            "original_claim":orig_claim,
                            "preprocessed_claim": cd_dict.get('preprocessed_claim','?'),
                            "ner_entities": cd_dict.get('ner_entities', []),
                            "factual_score": cd_dict.get('factual_score', 0.0),
                            "final_label":"Processing Error",
                            "confidence": 0.1,
                            "final_explanation":f"Worker error: {str(e)}",
                            "source": "Error"
                            }
                    try: self.claim_queue.task_done()
                    except ValueError: pass
        logging.info(f"W {t_name} finish.")

    # --- SHAP Function (unchanged) ---
    def train_and_run_shap(self, claims_processed_fully: list):
        if not self.shap_available:
            logging.warning("SHAP unavailable. Skipping SHAP analysis.")
            self.shap_explanations = [{"claim": cd.get('original_claim', '?'), "shap_values": "[SHAP Unavailable]"} for cd in claims_processed_fully]
            return
        if not claims_processed_fully:
            logging.info("SHAP: No claims went through the full pipeline, skipping SHAP.")
            self.shap_explanations = []
            return

        valid_claims_data = [cd for cd in claims_processed_fully if cd.get("original_claim") and isinstance(cd.get("original_claim"), str)]
        if not valid_claims_data:
            logging.warning("No valid claims data with text found for SHAP.")
            self.shap_explanations = []
            return

        logging.info(f"Attempting SHAP explanations for {len(valid_claims_data)} fully processed claims...");
        sentences = [cd['original_claim'] for cd in valid_claims_data]

        self.shap_explanations = [{"claim": cd['original_claim'], "shap_values": "[SHAP Pending]"} for cd in valid_claims_data]

        try:
            embed_dim = self.embedding_model.get_sentence_embedding_dimension()
        except Exception as e:
            logging.warning(f"Could not get embedding dimension for SHAP: {e}. Using fallback 384.")
            embed_dim = 384

        try:
            embeddings = self.embedding_model.encode(sentences, convert_to_tensor=False, show_progress_bar=False);
            if not isinstance(embeddings, np.ndarray) or embeddings.ndim != 2 or embeddings.shape[0] != len(sentences):
                 logging.error(f"SHAP: Invalid embeddings generated. Type: {type(embeddings)}, Shape: {getattr(embeddings, 'shape', 'N/A')}.")
                 self.shap_explanations = [{"claim": s, "shap_values": "[SHAP Embed Err]"} for s in sentences]; return
            logging.debug(f"SHAP embeddings shape: {embeddings.shape}")

            def predict_scores_for_shap(np_embeds):
                 if not isinstance(np_embeds, np.ndarray):
                     try:
                         np_embeds = np.array(np_embeds)
                         if np_embeds.ndim == 1: np_embeds = np_embeds.reshape(1, -1)
                         assert np_embeds.ndim == 2
                     except Exception as pred_e:
                          logging.error(f"SHAP predict input error: {pred_e}. Input type: {type(np_embeds)}")
                          num_inputs = len(np_embeds) if hasattr(np_embeds, '__len__') else 1
                          return np.full(num_inputs, 0.5)

                 scores = []
                 for e in np_embeds:
                     e_1d = e.flatten()
                     norm = float(np.linalg.norm(e_1d))
                     score = 0.5 + (norm / (norm * 25 + 1e-6)) if norm > 1e-6 else 0.5
                     scores.append(min(max(score, 0.0), 1.0))
                 return np.array(scores)

            logging.debug("Creating SHAP background data..."); bg_data=None
            n_samples = embeddings.shape[0]
            n_bg_samples = min(100, n_samples)
            n_clusters = min(10, n_bg_samples)

            try:
                 if n_samples >= n_clusters and n_clusters > 1:
                     bg_obj = shap.kmeans(embeddings, n_clusters)
                     if hasattr(bg_obj, 'data') and isinstance(bg_obj.data, np.ndarray): bg_data = bg_obj.data; logging.debug(f"Using KMeans background data, shape: {bg_data.shape}")
                     else: logging.warning("KMeans object missing '.data', using raw embeddings subset."); indices = np.random.choice(n_samples, n_bg_samples, replace=False); bg_data = embeddings[indices]
                 elif n_samples > 0: logging.debug(f"Using raw embeddings subset for background (samples: {n_samples})"); indices = np.random.choice(n_samples, n_bg_samples, replace=False); bg_data = embeddings[indices]
                 else: logging.error("Cannot create background data: No embeddings."); self.shap_explanations = [{"claim": s, "shap_values": "[SHAP BG Err]"} for s in sentences]; return
            except Exception as ke: logging.warning(f"SHAP KMeans failed: {ke}. Using raw embeddings subset."); indices = np.random.choice(n_samples, n_bg_samples, replace=False); bg_data = embeddings[indices]

            if bg_data is None or not isinstance(bg_data, np.ndarray) or bg_data.shape[0] == 0:
                logging.error("SHAP background data preparation failed or empty."); self.shap_explanations = [{"claim": s, "shap_values": "[SHAP BG Err]"} for s in sentences]; return

            if bg_data.ndim == 1: bg_data = bg_data.reshape(1,-1)
            logging.debug(f"Final background data shape: {bg_data.shape}")

            logging.debug("Initializing SHAP KernelExplainer...");
            explainer=shap.KernelExplainer(predict_scores_for_shap, bg_data)

            logging.info(f"Calculating SHAP values for {embeddings.shape[0]} instance(s)...");
            n_samples_shap = min(50, 2 * embeddings.shape[1] + 2048)
            shap_vals=explainer.shap_values(embeddings, nsamples=n_samples_shap)
            logging.info("SHAP values calculated.")

            calculated_explanations = []
            if isinstance(shap_vals, np.ndarray):
                expected_shape_single = (embed_dim,)
                expected_shape_multi = (len(sentences), embed_dim)
                if shap_vals.shape == expected_shape_multi:
                    for i, sentence in enumerate(sentences): calculated_explanations.append({"claim": sentence, "shap_values": shap_vals[i].tolist()})
                elif len(sentences) == 1 and shap_vals.shape == expected_shape_single:
                     calculated_explanations.append({"claim": sentences[0], "shap_values": shap_vals.tolist()})
                else:
                     logging.error(f"SHAP values shape mismatch. Got {shap_vals.shape}."); calculated_explanations = [{"claim": s, "shap_values": f"[SHAP Shape Err: {shap_vals.shape}]"} for s in sentences]
            elif isinstance(shap_vals, list) and len(shap_vals) == len(sentences):
                 if all(isinstance(item, np.ndarray) and item.shape == (embed_dim,) for item in shap_vals):
                      for i, sentence in enumerate(sentences): calculated_explanations.append({"claim": sentence, "shap_values": shap_vals[i].tolist()})
                 else: logging.error(f"SHAP values list content mismatch."); calculated_explanations = [{"claim": s, "shap_values": "[SHAP List Err]"} for s in sentences]
            else: logging.error(f"SHAP values unexpected type: {type(shap_vals)}."); calculated_explanations = [{"claim": s, "shap_values": f"[SHAP Type Err: {type(shap_vals).__name__}]"} for s in sentences]

            if calculated_explanations: logging.info(f"SHAP results stored for {len(calculated_explanations)} claims.")
            self.shap_explanations = calculated_explanations

        except Exception as e:
            logging.error(f"SHAP generation error: {e}", exc_info=True);
            self.shap_explanations = [{"claim": s, "shap_values": f"[SHAP Error: {type(e).__name__}]"} for s in sentences]

    # --- Chain of Thought and Check methods (unchanged from previous corrected version) ---
    def generate_chain_of_thought(self, all_processed_claims: list, non_checkable_claims: list) -> str:
        cot = ["Chain of Thought Summary:"]
        cot.append("1. Input Segmentation & Filtering:")
        total_initial = len(all_processed_claims) + len(non_checkable_claims)
        cot.append(f"   - Initial Sentences: {total_initial}")
        if non_checkable_claims: cot.append(f"   - Filtered Non-Checkable ({len(non_checkable_claims)}): {sorted(non_checkable_claims)}") # Sort for consistency
        cot.append(f"   - Checkable Claims Identified: {len(all_processed_claims)}")

        prioritized_claims = [c for c in all_processed_claims if c.get('source') != 'Knowledge Graph' and 'priority' in c]
        # Sort prioritized claims by original text for consistent CoT
        checkable_claims_str = [f"'{c.get('original_claim','?')[:50]}...' (Prio:{c.get('priority',0):.3f})"
                                for c in sorted(prioritized_claims, key=lambda x: x.get('original_claim', ''))]
        if checkable_claims_str: cot.append(f"   - Preprocessed & Prioritized ({len(checkable_claims_str)}): [{', '.join(checkable_claims_str)}]")
        elif len(all_processed_claims) > 0: cot.append("   - Preprocessing & Prioritization: Attempted, but details missing or only KG hits/errors.")

        kg_hits = [c for c in all_processed_claims if c.get('source') == 'Knowledge Graph']
        pipeline_processed = [c for c in all_processed_claims if c.get('source') == 'Full Pipeline']
        errors = [c for c in all_processed_claims if c.get('source') == 'Error']

        cot.append("2. Knowledge Graph Check & Processing:")
        if kg_hits: cot.append(f"   - KG Hits ({len(kg_hits)}): Found existing reliable verdicts.")
        else: cot.append("   - KG Hits: None found matching criteria.")
        if pipeline_processed: cot.append(f"   - Full Pipeline ({len(pipeline_processed)}): Claims processed via APIs and Synthesizing LLM.")
        else: cot.append("   - Full Pipeline: No claims required/completed full processing.")
        if errors: cot.append(f"   - Errors ({len(errors)}): Claims encountered processing errors (Preprocessing, Worker, or Missing Result).")

        cot.append("3. Processed Claim Results:")
        all_processed_claims_sorted = sorted(all_processed_claims, key=lambda x: x.get('original_claim', ''))
        for i, res in enumerate(all_processed_claims_sorted):
                claim = res.get('original_claim','?')
                source = res.get('source', '?')
                final_label = res.get('final_label', '?')
                confidence = res.get('confidence', 0.0)
                explanation = res.get('final_explanation', '?')

                cot.append(f"   - Claim {i+1}: '{claim}' (Source: {source})")
                if source == "Knowledge Graph":
                    cot.append(f"     - Final Verdict: {final_label} (Confidence: {confidence:.2f})")
                    cot.append(f"     - Explanation: {explanation}")
                elif source == "Full Pipeline":
                     cot.append(f"     - Initial Check: Verdict='{res.get('initial_verdict_raw','?')}'")
                     cot.append(f"     - RAG Status: {res.get('rag_status','?')}")
                     cot.append(f"     - LLM Final Verdict: {final_label} (Confidence: {confidence:.2f})")
                     cot.append(f"     - LLM Justification: {explanation}")
                     if self.neo4j_driver: cot.append(f"     - Neo4j Storage: Stored/Updated")
                elif source == "Error":
                     cot.append(f"     - Final Verdict: {final_label} (Confidence: {confidence:.2f})")
                     cot.append(f"     - Explanation: {explanation}")
                else:
                     cot.append(f"     - Status: Result Missing or Unknown Source!")

        cot.append("4. SHAP Analysis (for fully processed claims):")
        if self.shap_explanations:
             processed_claims_originals = {c['original_claim'] for c in pipeline_processed}
             relevant_explanations = [ex for ex in self.shap_explanations if ex.get('claim') in processed_claims_originals]

             if relevant_explanations:
                 sh_sum=[]; has_real_values=False; has_errors=False
                 relevant_explanations_sorted = sorted(relevant_explanations, key=lambda x: x.get('claim', ''))
                 for ex in relevant_explanations_sorted:
                     v=ex.get('shap_values',[]); s="[Err/Unavail]"
                     if isinstance(v,list) and v:
                         is_numeric = all(isinstance(x,(int,float)) for x in v)
                         is_error_fallback = any(isinstance(val, str) and val.startswith('[SHAP') for val in v)
                         is_all_zero = all(abs(float(x)) < 1e-9 for x in v if isinstance(x,(int,float)))
                         if is_numeric and not is_error_fallback and not is_all_zero: s=f"[...{len(v)} SHAP vals...]"; has_real_values=True
                         elif is_all_zero and not is_error_fallback: s="[Zero Values]"
                         elif is_error_fallback: s = v[0] if v and isinstance(v[0], str) else "[SHAP Err Fallback]"; has_errors = True
                         else: s="[Mixed/Invalid Data]"
                     elif isinstance(v,str) and v.startswith('[SHAP'): s=v; has_errors = True
                     elif not v: s="[No Data]"
                     sh_sum.append(f"'{ex.get('claim','?')[:40]}...': {s}")

                 status = "Generated values." if has_real_values else ("Failed/Unavailable." if has_errors else "Zero values reported.")
                 cot.append(f"   - SHAP Status: {status} Details: {{{', '.join(sh_sum)}}}")
             else:
                 cot.append("   - SHAP analysis skipped (no relevant claims processed via pipeline or SHAP failed early).")
        else:
            cot.append("   - SHAP analysis skipped or no results structure.")

        return "\n".join(cot)

    def check(self, text: str, num_workers: int = 2) -> dict:
        start=time.time(); logging.info(f"Starting check: \"{text[:100]}...\"")
        with self.results_lock:
            self.results={}; self.shap_explanations=[]; self.raw_fact_checks={}; self.raw_searches={}
        while True:
            try: self.claim_queue.get_nowait(); self.claim_queue.task_done()
            except queue.Empty: break
            except Exception as e: logging.warning(f"Error clearing queue item: {e}")

        logging.info("Step 1: Preprocessing & Filtering...");
        checkable_claims_initial_data, non_checkable_sents=self.preprocess_and_filter(text)

        # Store non-checkable claims immediately if needed, or just keep the list
        # final_non_checkable = [{"original_claim": s, "source": "Filtered"} for s in non_checkable_sents]

        if not checkable_claims_initial_data:
            logging.warning("No checkable claims found after preprocessing.");
            return {"processed_claims":[], "non_checkable_claims":non_checkable_sents, "summary":"No checkable claims found.", "raw_fact_checks":{}, "raw_searches":{}, "shap_explanations": []}

        logging.info("Step 2: Prioritizing Checkable Claims...");
        prioritized_claims_data=self.classify_and_prioritize_claims(checkable_claims_initial_data)
        if not prioritized_claims_data:
             logging.warning("Prioritization failed or returned empty list.");
             prioritized_claims_data = checkable_claims_initial_data # Fallback
             if not prioritized_claims_data:
                return {"processed_claims":[], "non_checkable_claims":non_checkable_sents, "summary":"Prioritization failed and no claims available.", "raw_fact_checks":{}, "raw_searches":{}, "shap_explanations": []}

        logging.info("Step 3: Checking Knowledge Graph (Neo4j)...");
        claims_to_process_fully = []
        claims_found_in_kg = []
        claims_with_preprocessing_error = [] # Track claims failing preprocessing

        for claim_data_dict in prioritized_claims_data:
             if not isinstance(claim_data_dict, dict):
                  logging.error(f"Invalid item in prioritized claims data: {claim_data_dict}. Skipping.")
                  continue
             preprocessed_claim_text = claim_data_dict.get('preprocessed_claim')
             original_claim = claim_data_dict.get('original_claim', '?')

             # Check if preprocessing seems to have failed (empty or identical without NLP)
             preprocessing_failed = not preprocessed_claim_text or \
                                    (preprocessed_claim_text == original_claim and not self.nlp_available)

             if preprocessing_failed:
                  logging.warning(f"Marking claim as Preprocessing Error: '{original_claim}'")
                  claim_data_dict['source'] = 'Error'
                  claim_data_dict['final_label'] = 'Preprocessing Error'
                  claim_data_dict['confidence'] = 0.1
                  claim_data_dict['final_explanation'] = 'Failed to generate distinct preprocessed text (or NLP unavailable).'
                  claims_with_preprocessing_error.append(claim_data_dict)
                  continue # Don't check KG or queue

             kg_result = self.check_kg_for_claim(preprocessed_claim_text)
             if kg_result:
                 claims_found_in_kg.append(kg_result)
             else:
                 claims_to_process_fully.append(claim_data_dict)

        logging.info(f"KG Check Results: Found={len(claims_found_in_kg)}, Needs Full Processing={len(claims_to_process_fully)}, Preprocessing Errors={len(claims_with_preprocessing_error)}")

        if claims_to_process_fully:
            logging.info("Step 4a: Queueing claims for full processing...");
            self.add_claims_to_queue(claims_to_process_fully)

            logging.info("Step 4b: Processing via Workers...");
            threads=[]; n_cpu=os.cpu_count() or 1;
            n_workers = min(num_workers, self.claim_queue.qsize(), n_cpu if n_cpu else 1)
            if n_workers > 0:
                logging.info(f"Starting {n_workers} workers...");
                for i in range(n_workers):
                    t=Thread(target=self.worker,name=f"Worker-{i+1}",daemon=True);
                    t.start();
                    threads.append(t)
                self.claim_queue.join(); logging.info("Workers finished processing queue.")
                for t in threads:
                    t.join(timeout=10.0)
                    if t.is_alive(): logging.warning(f"Thread {t.name} still alive after join!")
            else:
                 logging.info("Worker processing skipped (queue empty or num_workers=0).")
        else:
            logging.info("Step 4: Skipping full pipeline processing.")

        logging.info("Step 5: Generating SHAP (if installed)...");
        self.train_and_run_shap(claims_to_process_fully)

        logging.info("Step 6: Consolidating final results...");
        final_results_list = []
        processed_count = 0
        error_count = 0

        final_results_list.extend(claims_found_in_kg)
        processed_count += len(claims_found_in_kg)
        final_results_list.extend(claims_with_preprocessing_error)
        processed_count += len(claims_with_preprocessing_error)
        error_count += len(claims_with_preprocessing_error)

        with self.results_lock:
            for cd in claims_to_process_fully:
                 original_claim = cd.get('original_claim')
                 if not original_claim: continue
                 pipeline_result = self.results.get(original_claim)
                 if pipeline_result:
                      if not any(res.get('original_claim') == original_claim for res in final_results_list):
                           final_results_list.append(pipeline_result)
                           processed_count += 1
                           if pipeline_result.get("source") == "Error" or "Error" in pipeline_result.get("final_label",""): error_count += 1
                 else:
                      logging.error(f"Result missing from self.results for fully processed claim: '{original_claim}'. Adding error entry.")
                      final_results_list.append({
                          "original_claim": original_claim, "preprocessed_claim": cd.get('preprocessed_claim','?'), "ner_entities": cd.get('ner_entities', []), "factual_score": cd.get('factual_score', 0.0),
                          "final_label": "Missing Result", "confidence": 0.0, "final_explanation": "Result lost after worker processing.", "source": "Error"
                          })
                      processed_count += 1; error_count += 1

        summary = self.generate_chain_of_thought(final_results_list, non_checkable_sents)

        duration = time.time() - start
        fully_processed_ok_count = len([r for r in final_results_list if r.get('source') == 'Full Pipeline'])
        logging.info(f"Check complete. Initial Checkable={len(checkable_claims_initial_data)}, KG Hits={len(claims_found_in_kg)}, Pipeline OK={fully_processed_ok_count}, Total Errors={error_count} in {duration:.2f}s.")

        return {
            "processed_claims": final_results_list, "non_checkable_claims": non_checkable_sents, "summary": summary,
            "raw_fact_checks": self.raw_fact_checks, "raw_searches": self.raw_searches, "shap_explanations": self.shap_explanations
            }

    def close_neo4j(self):
        """Closes the Neo4j driver connection."""
        if self.neo4j_driver:
            try: self.neo4j_driver.close(); logging.info("Neo4j driver closed.")
            except Exception as e: logging.error(f"Error closing Neo4j driver: {e}")

# --- Main Execution Block (unchanged) ---
if __name__ == "__main__":
    if shap is None: print("\nWARN: SHAP lib missing. Run: pip install shap\n")
    else: print("SHAP library found.")
    if spacy is None: print("\nWARN: spaCy lib missing. Run: pip install spacy\n")
    elif NLP is None: print("\nWARN: spaCy model 'en_core_web_sm' missing. Run: python -m spacy download en_core_web_sm\n")
    else: print("spaCy library and model found.")
    if Groq is None: print(f"\nWARN: Groq lib missing. LLM ({LLM_PROVIDER}) disabled. Run: pip install groq\n")
    elif groq_client is None: print(f"\nWARN: Groq client init failed (check API key/env). LLM ({LLM_PROVIDER}) disabled.\n")
    else: print(f"{LLM_PROVIDER} client available (Model: {LLM_MODEL_NAME}).")

    print("Fact Checker Initializing...")
    if not GOOGLE_API_KEY or not GOOGLE_CSE_ID:
        logging.critical("CRITICAL: Google API Key or CSE ID missing."); print("\nCRITICAL ERROR: Google API Key or CSE ID missing. Exiting."); exit(1)
    else: logging.info("Google keys loaded.")
    if not NEO4J_PASSWORD:
        logging.critical("CRITICAL: NEO4J_PASSWORD missing."); print("\nCRITICAL ERROR: Neo4j password missing. Exiting."); exit(1)
    else: logging.info("Neo4j password loaded.")

    checker = None
    try:
        checker = FactChecker()
        if checker.neo4j_driver is None: print("\nCRITICAL ERROR: Neo4j driver failed. Exiting."); exit(1)
        print("Fact Checker Initialized Successfully.")
    except RuntimeError as e: logging.critical(f"Initialization failed (Model Loading?): {e}",exc_info=True); print(f"\nCRITICAL ERROR: Init failed. Logs: {log_file}. Error: {e}"); exit(1)
    except Exception as e: logging.critical(f"Unexpected initialization error: {e}",exc_info=True); print(f"\nCRITICAL UNEXPECTED ERROR during init. Logs: {log_file}. Error: {e}"); exit(1)

    input_text = (
        "The Eiffel Tower is located in Berlin. Fact checkers say this is false. "
        "I think Paris is the most beautiful city. COVID-19 vaccines are ineffective according to some studies. "
        "Water boils at 100 degrees Celsius at sea level. This statement is true. "
        "The earth is flat according to some sources. Llamas are native to North America, not South America. "
        "Quantum computing will break all encryption soon. This sentence should be ignored."
        "Is VeriStream the best fact-checker?"
    )
    print(f"\nInput Text:\n{input_text}\n")

    print("\n--- Starting Fact Check Pipeline ---\n")
    if not checker: print("Checker not initialized. Exiting."); exit(1)

    try:
        results_data = checker.check(input_text, num_workers=2)
    except Exception as check_e:
        logging.critical(f"Critical error during checker.check(): {check_e}", exc_info=True)
        print(f"\nCRITICAL ERROR during fact checking process. Check logs: {log_file}. Error: {check_e}")
        if checker: checker.close_neo4j()
        exit(1)

    # --- OUTPUT ORDER (No changes needed in printing logic) ---
    print("\n" + "="*25 + " Intermediate Output 1: Raw Google Fact Check API Results " + "="*15)
    raw_checks = results_data.get("raw_fact_checks", {})
    if raw_checks:
        print(" (Note: Shows results only for claims requiring full API processing)")
        for claim in sorted(raw_checks.keys()):
            api_res_list = raw_checks[claim]
            print(f"\nClaim (Original): \"{claim}\"")
            if api_res_list and isinstance(api_res_list, list):
                for res_item in api_res_list: print(f"  - Verdict: {res_item.get('verdict','?')} | Evidence: {res_item.get('evidence','?')}")
            else: print("  - No result stored or invalid format.")
    else: print("  - No Fact Check API calls were made or stored.")
    print("="*81)

    SHOW_RAW_SEARCH = False
    raw_searches = results_data.get("raw_searches", {})
    if SHOW_RAW_SEARCH and raw_searches:
        print("\n" + "="*25 + " Intermediate Output 2: Raw Google Custom Search Snippets " + "="*14)
        print(" (Note: Shows results only for claims requiring full API processing)")
        for claim in sorted(raw_searches.keys()):
            search_data = raw_searches[claim]
            print(f"\nClaim (Original): \"{claim}\""); print(f"  (API Query: \"{search_data.get('query', '?')}\")")
            search_results = search_data.get("results", [])
            if search_results:
                for i, item in enumerate(search_results): print(f"  {i+1}. T: {item.get('title','?')}\n     S: {item.get('snippet','?')}\n     L: {item.get('link','?')}")
            elif search_data.get("response") is not None: print("  - Search OK, no items returned.")
            else: print("  - Search API call likely failed or response missing.")
        print("="*81)

    print("\n" + "="*25 + " Preprocessing Output: Filtered Non-Checkable Sentences " + "="*16)
    non_checkable = results_data.get("non_checkable_claims", [])
    if non_checkable:
        for i, claim in enumerate(sorted(non_checkable)): print(f"  {i+1}. \"{claim}\"")
    else: print("  - No sentences were filtered out as non-checkable.")
    print("="*81)

    print("\n" + "="*30 + " Final Processed Claim Details " + "="*30)
    processed_claims = results_data.get("processed_claims", [])
    if processed_claims:
        sorted_results = sorted(processed_claims, key=lambda x: x.get('original_claim', ''))
        for i, res in enumerate(sorted_results):
            source = res.get('source', 'Unknown')
            final_label = res.get('final_label', 'N/A')
            confidence = res.get('confidence', 0.0)
            explanation = res.get('final_explanation', 'N/A')
            original_claim = res.get('original_claim', '?')
            preprocessed_claim = res.get('preprocessed_claim', 'N/A')
            factual_score = res.get('factual_score')
            initial_verdict = res.get('initial_verdict_raw', '?')
            rag_status = res.get('rag_status', '?')
            ner_entities = res.get('ner_entities', [])
            top_snippets = res.get('top_rag_snippets', [])
            kg_timestamp = res.get('kg_timestamp')

            print(f"\nClaim {i+1} (Original): \"{original_claim}\" [Source: {source}]")
            print(f"  - Preprocessed: \"{preprocessed_claim}\"")

            if source == "Full Pipeline":
                if ner_entities: print("  - NER Entities: {}".format(', '.join(["{}({})".format(e['text'], e['label']) for e in ner_entities])))
                else: print("  - NER Entities: None Found")
                print(f"  - Factual Score (0-1): {factual_score:.3f}" if factual_score is not None else "N/A")
                print(f"  - Initial Check Result: '{initial_verdict}'")
                print(f"  - RAG Status: {rag_status}")
                if top_snippets:
                    print("  - Top RAG Snippets:");
                    for j, snip in enumerate(top_snippets): print(f"    {j+1}. {snip}")
                else: print("  - Top RAG Snippets: None")
                print(f"  - Final Verdict (Synthesized): {final_label} (Confidence: {confidence:.2f})") # Indicate synthesized
                print(f"  - LLM Justification: {explanation}")
                if checker and checker.neo4j_driver: print(f"  - Neo4j Storage: Stored/Updated in DB '{NEO4J_DATABASE}'")

            elif source == "Knowledge Graph":
                print(f"  - Final Verdict (From KG): {final_label} (Confidence: {confidence:.2f})")
                print(f"  - KG Explanation: {explanation}")
                if kg_timestamp: print(f"  - KG Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime(kg_timestamp))}" if kg_timestamp else "N/A")

            elif source == "Error":
                print(f"  - Final Verdict: {final_label} (Confidence: {confidence:.2f})")
                print(f"  - Explanation: {explanation}")
            else:
                 print(f"  - Status: Unknown processing source or incomplete result format.")
                 print(f"  - Details: Label={final_label}, Conf={confidence:.2f}, Explain={explanation}")

    else: print("  - No checkable claims were processed or results available.")
    print("="*83)

    print("\n" + "="*30 + " XAI (SHAP) Summary " + "="*37)
    shap_explanations = results_data.get("shap_explanations", [])
    if shap_explanations:
         fully_processed_claims_texts = {p['original_claim'] for p in processed_claims if p.get('source') == 'Full Pipeline'}
         relevant_explanations = [ex for ex in shap_explanations if ex.get('claim') in fully_processed_claims_texts]
         if relevant_explanations:
             print(f" (Note: Shows results only for {len(relevant_explanations)} claim(s) requiring full API processing)")
             shap_summary=[]; has_real_values=False; has_errors=False
             relevant_explanations_sorted = sorted(relevant_explanations, key=lambda x: x.get('claim', ''))
             for expl in relevant_explanations_sorted:
                 claim_text = expl.get('claim', '?')
                 v = expl.get('shap_values', []); s="[Err/Unavail]"
                 if isinstance(v,list) and v:
                     is_numeric = all(isinstance(x,(int,float)) for x in v)
                     is_error_fallback = any(isinstance(val, str) and val.startswith('[SHAP') for val in v)
                     is_all_zero = all(abs(float(x)) < 1e-9 for x in v if isinstance(x,(int,float)))
                     if is_numeric and not is_error_fallback and not is_all_zero: s=f"[...{len(v)} values]"; has_real_values=True
                     elif is_all_zero and not is_error_fallback: s="[Zero Values]"
                     elif is_error_fallback: s = v[0] if v and isinstance(v[0], str) else "[SHAP Err Fallback]"; has_errors = True
                     else: s="[Mixed/Invalid Data]"
                 elif isinstance(v,str) and v.startswith('[SHAP'): s=v; has_errors = True
                 elif not v: s="[No Data]"
                 shap_summary.append(f"'{claim_text[:40]}...': {s}")
             status = "Generated values." if has_real_values else ("Failed/Unavailable." if has_errors else "Zero values reported.")
             print(f"  - SHAP Status: {status}")
             if shap_summary: print(f"  - Details: {{{', '.join(shap_summary)}}}")
             if has_errors: print(f"\n  *** SHAP Error Detected: Check '{log_file}' for details. ***")
             elif not has_real_values and relevant_explanations: print(f"\n  *** SHAP produced zero values or failed: Prediction function may be unsuitable or error occurred. Check '{log_file}'. ***")
         else: print("  - SHAP analysis not applicable (no claims required full processing, SHAP failed early, or results missing).")
    elif shap is None: print("  - SHAP library not installed.")
    else: print("  - SHAP analysis results structure missing.")
    print("="*86)

    print("\n" + "="*30 + " Chain of Thought Summary " + "="*30)
    print(results_data.get("summary", "No summary generated."))
    print("="*86)
    print(f"\nLog file generated at: {log_file}")

    if checker: checker.close_neo4j()
    print("\nScript finished.")