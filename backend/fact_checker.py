# backend/fact_checker.py
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
NUM_SEARCH_RESULTS = 5
RAG_K = 5
# Keywords for filtering non-checkable claims
OPINION_PHRASES = ["i think", "i believe", "in my opinion", "seems like", "feels like", "should be", "must be"]
SUBJECTIVE_ADJECTIVES = ["beautiful", "ugly", "amazing", "terrible", "wonderful", "awful", "best", "worst", "nice", "bad", "good", "great"] # Added great
SELF_REFERENCE_WORDS = ["this sentence", "this claim", "this statement", "i say", "i state", "this phrase"]
KG_RELEVANT_NER_LABELS = {"PERSON", "ORG", "GPE", "LOC", "DATE", "TIME", "MONEY", "QUANTITY", "PERCENT", "CARDINAL", "ORDINAL", "PRODUCT", "EVENT", "WORK_OF_ART", "LAW", "NORP", "FAC"}

# Logging
log_file = 'fact_checker.log'
# Clear log file at start
with open(log_file, 'w', encoding='utf-8') as f: pass
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(threadName)s] - %(message)s', handlers=[logging.FileHandler(log_file, encoding='utf-8'), logging.StreamHandler(stream=sys.stdout)])

# NLTK Downloads
try: nltk.data.find('tokenizers/punkt')
except LookupError: logging.info("Downloading NLTK 'punkt'..."); nltk.download('punkt', quiet=True); logging.info("'punkt' downloaded.")

try: nltk.data.find('tokenizers/punkt_tab')
except LookupError: logging.info("Downloading NLTK 'punkt_tab'..."); nltk.download('punkt_tab', quiet=True); logging.info("'punkt_tab' downloaded.")

# --- API Functions ---
def google_fact_check(query: str, api_key: str) -> list:
    """Queries Google Fact Check API using the provided query string."""
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
    """Queries Google Custom Search API using the provided query string."""
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


# --- *** REVISED LLM Final Verdict Function (v2) *** ---
def get_llm_final_verdict_v2(
    claim: str, # Use original claim
    initial_check_verdict: str,
    initial_check_evidence: str,
    rag_evidence: list,
    rag_status_msg: str
) -> dict:
    """
    Uses Groq LLM to determine a final verdict, confidence, and justification
    by synthesizing the initial check results, RAG evidence, AND its own internal knowledge.
    """
    if not groq_client:
        return {"final_label": "LLM Error", "confidence": 0.1, "explanation": "Groq client not initialized."}

    # --- Format Inputs for Prompt ---
    # Initial Check Info
    initial_check_info = "Initial Fact-Check API Data:\n"
    if initial_check_verdict and initial_check_verdict != "Error" and initial_check_verdict != "N/A":
        initial_check_info += f"- Verdict: {initial_check_verdict}\n"
        initial_check_info += f"- Evidence Source: {initial_check_evidence}\n"
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

    # --- Revised Prompt for Synthesis (v2 - includes internal knowledge) ---
    prompt = f"""You are an expert fact-checker and critical thinker. Your goal is to determine the most accurate final verdict (True/False) and confidence level for the 'Claim' by considering THREE sources of information:
1.  **Your Internal Knowledge:** Assess the claim based on your general knowledge and reasoning abilities.
2.  **Initial Fact-Check API Data:** Evaluate the preliminary verdict and source provided (if available and reliable).
3.  **Retrieved Web Snippets (RAG):** Analyze the supporting/contradicting evidence from web search results.

Claim: "{claim}"

{initial_check_info}
{rag_info}

**Synthesis Task:**
1.  **Evaluate the Claim:** Based *only* on your internal knowledge, is the claim likely True or False?
2.  **Evaluate External Evidence:**
    *   Does the Initial Check provide a clear, relevant verdict? Is its source credible for *this specific claim*? (e.g., a health site commenting on physics might be less credible).
    *   Do the RAG snippets directly support, contradict, or are they irrelevant to the claim? How consistent, relevant, and reliable do the RAG snippets seem?
3.  **Synthesize and Weigh:** Combine your internal assessment with the external evidence.
    *   If your knowledge strongly contradicts weak/irrelevant/conflicting external evidence, prioritize your knowledge.
    *   If external evidence (especially strong RAG) convincingly supports/contradicts the claim, give it significant weight, even if it counters your initial assessment (but double-check RAG relevance).
    *   If Initial Check and RAG conflict, which one seems more reliable/relevant for this claim?
    *   If all sources conflict or are weak/irrelevant, acknowledge the uncertainty.
4.  **Determine Final Verdict (True/False):** Based on your *overall synthesis*:
    *   "True": If the combined evidence (including your knowledge) strongly and consistently supports the claim.
    *   "False": If the combined evidence strongly contradicts the claim OR if the overall evidence (knowledge + external) is insufficient, ambiguous, or conflicting, preventing verification. **If verification is not possible, the verdict is False.**
5.  **Assign Confidence Score (0.0 - 1.0):** Reflect your certainty based on the *strength, consistency, and agreement* across *all three sources* (internal knowledge, initial check, RAG):
    *   *High Confidence (> 0.85):* Strong agreement between reliable external evidence and internal knowledge. Or, very strong, consistent RAG/Initial Check that overrides a conflicting weaker source.
    *   *Medium Confidence (0.6 - 0.85):* Some conflict or ambiguity. Maybe strong internal knowledge + weak/mixed external evidence, or vice-versa. RAG partially supporting/contradicting.
    *   *Low Confidence (<= 0.6):* Significant conflict between sources, OR all sources (internal knowledge + external) are weak/irrelevant/missing. **A 'False' verdict due to overall lack of evidence warrants low confidence.**
6.  **Provide Justification:** Explain your reasoning in 1-2 sentences. Explicitly state how your internal knowledge, the Initial Check, and RAG evidence contributed to the final verdict and confidence. Mention if one source strongly influenced the outcome over others. (e.g., "Internal knowledge confirms water boils at 100C at sea level; the 'False' Initial Check was disregarded as irrelevant, and RAG provided only tangentially related info. Verdict True based on knowledge.", "Initial check and RAG strongly contradicted the claim, aligning with internal knowledge.", "Internal knowledge suggested False, RAG was irrelevant, Initial check was Unknown. Verdict False due to lack of supporting evidence.")

**Output Format:** Respond with *only* a valid JSON object containing exactly three keys: "verdict" (string: "True" or "False"), "confidence" (float: 0.0-1.0), "justification" (string).

Example (Internal Knowledge + RAG overrides bad Initial):
{{
  "verdict": "True",
  "confidence": 0.95,
  "justification": "My internal knowledge confirms water boils at 100°C at sea level. The 'False' Initial Check was likely irrelevant or erroneous for this claim. RAG snippets, while about boiling water advisories, don't contradict the fact. Verdict based primarily on established scientific knowledge."
}}

Example (RAG overrides Initial and weak internal):
{{
  "verdict": "False",
  "confidence": 0.8,
  "justification": "While internal knowledge might be uncertain, the Initial Check ('Distorts Facts') and multiple relevant RAG snippets consistently contradicted the claim's assertion about vaccine ineffectiveness. Verdict based on stronger external evidence."
}}

Example (All sources weak/irrelevant):
{{
  "verdict": "False",
  "confidence": 0.3,
  "justification": "Internal knowledge cannot verify this specific claim. The Initial Check was 'Unknown' and the RAG snippets were irrelevant. Verdict is 'False' due to insufficient evidence from all sources."
}}

Your JSON Response:
"""
    logging.debug(f"LLM Synthesized Verdict Prompt (v2) for '{claim}':\n{prompt}")
    try:
        response = groq_client.chat.completions.create(
            model=LLM_MODEL_NAME,
            messages=[
                {"role": "system", "content": "Output only valid JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1, # Lower temperature for more deterministic reasoning
            max_tokens=350 # Allow slightly more tokens for complex justification
        )
        llm_output_text = response.choices[0].message.content
        logging.debug(f"Groq LLM Raw (v2): {llm_output_text}")

        # --- Robust JSON Parsing ---
        try:
            llm_output_text_clean = llm_output_text.strip()
            # Remove potential markdown code blocks
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

            # Validate structure and types
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

    # --- Handle API Errors ---
    except APIConnectionError as e: logging.error(f"Groq ConnErr for '{claim[:50]}...': {e}"); return {"final_label": "LLM Error", "confidence": 0.1, "explanation": f"API Conn Err: {e}"}
    except RateLimitError as e: logging.error(f"Groq RateLimit for '{claim[:50]}...': {e}"); time.sleep(5); return {"final_label": "LLM Error", "confidence": 0.1, "explanation": f"API Rate Limit: {e}"}
    except APIStatusError as e: logging.error(f"Groq API Status {e.status_code} for '{claim[:50]}...': {e.response}"); return {"final_label": "LLM Error", "confidence": 0.1, "explanation": f"API Status {e.status_code}"}
    except Exception as e: logging.error(f"Unexpected Groq API err for '{claim[:50]}...': {e}", exc_info=True); return {"final_label": "LLM Error", "confidence": 0.1, "explanation": f"API Err: {e}"}
# --- End Revised LLM Function (v2) ---


# --- Helper Functions ---
def is_claim_checkable(sentence: str) -> bool:
    """Checks if a sentence appears to be a checkable factual claim."""
    # No changes needed here based on requirements
    if not sentence or not isinstance(sentence, str): return False
    text_lower = sentence.lower().strip()
    # Remove surrounding quotes for checks
    if text_lower.startswith('"') and text_lower.endswith('"'): text_lower = text_lower[1:-1]
    if text_lower.startswith("'") and text_lower.endswith("'"): text_lower = text_lower[1:-1]
    # Check for opinion phrases
    for phrase in OPINION_PHRASES:
        if text_lower.startswith(phrase+" "): logging.info(f"Filtered (Opinion): '{sentence}'"); return False
    # Check for self-reference
    if any(word in text_lower for word in SELF_REFERENCE_WORDS): logging.info(f"Filtered (Self-Ref): '{sentence}'"); return False
    # Check for questions
    if text_lower.endswith("?"): logging.info(f"Filtered (Question): '{sentence}'"); return False
    # Use NLP if available for deeper checks
    if NLP:
        try:
            doc = NLP(sentence)
            # Check for subjective adjectives
            if any(t.lemma_.lower() in SUBJECTIVE_ADJECTIVES for t in doc if t.pos_=="ADJ"): logging.info(f"Filtered (Subjective): '{sentence}'"); return False
            # Check for basic sentence structure (avoid fragments if possible)
            has_verb = any(t.pos_ in ("VERB","AUX") for t in doc); has_subj = any(t.dep_ in ("nsubj","nsubjpass","csubj","csubjpass","expl") for t in doc)
            # If no verb or subject and very short, likely not a claim
            if not (has_verb or has_subj) and len(doc)<5: logging.info(f"Filtered (Structure): '{sentence}'"); return False
            # Check for imperatives (commands)
            if len(doc)>0 and doc[0].pos_=="VERB" and doc[0].tag_=="VB":
                 is_imperative = not any(t.dep_.startswith("nsubj") for t in doc) # Imperatives lack subjects
                 if is_imperative: logging.info(f"Filtered (Imperative): '{sentence}'"); return False
        except Exception as nlp_e:
            logging.error(f"spaCy processing error during checkability for '{sentence}': {nlp_e}")
            return False # Err on side of caution
    # Basic length check if NLP is not available
    elif len(text_lower.split()) < 3:
         logging.info(f"Filtered (Short/No NLP): '{sentence}'"); return False
    # If none of the above filters matched, assume it's checkable
    return True

def preprocess_claim_for_kg(original_claim: str) -> str:
    """
    Simplifies a claim primarily for matching against existing claims in the Knowledge Graph.
    Uses NLP (spaCy) if available to extract core components (Subject-Verb-Object/Complement).
    Returns the original claim if preprocessing fails or is deemed unnecessary.
    """
    if not original_claim: return original_claim
    if not NLP:
        logging.warning(f"NLP unavailable, using original claim for KG key: '{original_claim}'")
        return original_claim.strip()

    try:
        doc = NLP(original_claim); simplified_parts = []; root = None; subj = None; obj_or_comp = None; negation = False

        # Find the root and check for negation
        for token in doc:
            if token.dep_ == "ROOT": root = token
            if token.dep_ == "neg": negation = True

        if not root:
             # Fallback for very short sentences or parsing issues
             if len(doc) < 5: return original_claim.strip()
             else: logging.warning(f"No ROOT found for KG preprocessing of '{original_claim}', using original."); return original_claim.strip()

        # Find subject(s)
        subjects = [token for token in doc if "subj" in token.dep_]
        if subjects: subj = subjects[0] # Take the first subject found

        # Find object or complement connected to the root
        for child in root.children:
            # Looking for direct object, attribute, adjectival complement etc.
            if "obj" in child.dep_ or "attr" in child.dep_ or "acomp" in child.dep_:
                 obj_or_comp = child
                 # Try to find the noun/adj/num head within the object phrase if complex
                 while True:
                      potential_heads = [c for c in obj_or_comp.children if c.pos_ in ("NOUN", "PROPN", "ADJ", "NUM")]
                      if potential_heads: obj_or_comp = potential_heads[0]; continue
                      else: break # Stop if no clearer head found
                 break # Take the first object/complement found

        # Build simplified claim: Subject + (not) + Verb Lemma + Object/Complement Text
        if subj: simplified_parts.append(subj.text)
        if negation: simplified_parts.append("not")

        # Find main verb (usually root, sometimes head of root or auxiliary)
        main_verb = root
        if root.pos_ != 'VERB' and root.head.pos_ == 'VERB': main_verb = root.head # e.g., aux like "is located" -> use locate
        elif root.pos_ != 'VERB':
             # Sometimes root is noun/adj, check children for aux verb
             aux_verb = next((t for t in root.children if t.pos_ == 'AUX'), None)
             if aux_verb: main_verb = aux_verb
             # else stick with the root even if not a verb (might be nominal predicate)

        simplified_parts.append(main_verb.lemma_) # Use lemma for verb normalization

        if obj_or_comp: simplified_parts.append(obj_or_comp.text)

        simplified_claim = " ".join(simplified_parts).strip()

        # Ensure simplification didn't fail badly
        if len(simplified_claim.split()) < 2 or not simplified_claim:
            logging.warning(f"KG Preprocessing resulted in short/empty string ('{simplified_claim}') for '{original_claim}', using original for KG key.");
            return original_claim.strip()

        logging.info(f"Simplified '{original_claim}' -> '{simplified_claim}' (for KG Matching).");
        return simplified_claim

    except Exception as e:
        logging.error(f"Error preprocessing '{original_claim}' for KG: {e}", exc_info=True)
        return original_claim.strip() # Fallback to original on any error

# --- FactChecker Class ---
class FactChecker:
    def __init__(self):
        logging.info(f"Device set to use {DEVICE}")
        try:
            logging.info(f"Loading SentenceTransformer: {EMBEDDING_MODEL_NAME}");
            self.embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME, device=DEVICE)
            self.langchain_embeddings = SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL_NAME, model_kwargs={'device': DEVICE})
            logging.info("Embedding models loaded successfully.")
        except Exception as e:
            logging.error(f"Fatal: Embedding model loading failed: {e}", exc_info=True);
            raise RuntimeError(f"Could not load embedding model '{EMBEDDING_MODEL_NAME}': {e}") from e

        if shap is None: logging.warning("SHAP library not installed. Explainability features disabled.")
        self.shap_available = shap is not None
        if NLP is None: logging.warning("spaCy model not loaded. Claim filtering/KG preprocessing will be basic.")
        self.nlp_available = NLP is not None

        # Runtime data structures
        self.claim_queue = queue.Queue()
        self.results = {}
        self.results_lock = Lock()
        self.shap_explanations = []
        self.raw_fact_checks = {} # Store raw API responses
        self.raw_searches = {}    # Store raw search results

        # --- Neo4j Driver Initialization ---
        self.neo4j_driver = None
        if not NEO4J_URI or not NEO4J_USER or not NEO4J_PASSWORD:
            logging.error("Neo4j connection details (URI, USER, PASSWORD) missing in environment variables. KG features disabled.")
        else:
            try:
                self.neo4j_driver = GraphDatabase.driver(NEO4J_URI, auth=basic_auth(NEO4J_USER, NEO4J_PASSWORD))
                # Verify connection during initialization
                with self.neo4j_driver.session(database=NEO4J_DATABASE) as session:
                    session.run("RETURN 1") # Simple query to check connectivity and auth
                logging.info(f"Neo4j driver initialized successfully for database '{NEO4J_DATABASE}'.")
            except neo4j_exceptions.AuthError as auth_err:
                 logging.error(f"Neo4j Authentication Failed for user '{NEO4J_USER}'. Check credentials. KG features disabled. Error: {auth_err}")
                 self.neo4j_driver = None
            except neo4j_exceptions.ServiceUnavailable as conn_err:
                 logging.error(f"Neo4j Service Unavailable at URI '{NEO4J_URI}'. Check if Neo4j is running. KG features disabled. Error: {conn_err}")
                 self.neo4j_driver = None
            except Exception as e: # Catch any other driver init errors
                logging.error(f"Failed to initialize Neo4j driver for URI '{NEO4J_URI}', DB '{NEO4J_DATABASE}'. KG features disabled. Error: {e}", exc_info=True)
                self.neo4j_driver = None
        # --- End Neo4j Init ---

    def check_kg_for_claim(self, preprocessed_claim: str) -> dict | None:
        """
        Queries Neo4j using the *preprocessed* claim text to find a matching Claim node
        with a reliable verdict (above KG_CONFIDENCE_THRESHOLD).
        Returns the stored result dictionary if found, otherwise None.
        """
        if not self.neo4j_driver:
            # logging.warning("KG Check skipped: Neo4j driver not available.") # Logged once at init
            return None
        if not preprocessed_claim:
             logging.warning("KG Check skipped: Empty preprocessed claim provided.")
             return None

        logging.debug(f"KG Check: Querying for preprocessed claim: '{preprocessed_claim}'")
        # Query uses the preprocessed_text property which should be unique or indexed
        query = """
        MATCH (c:Claim {preprocessed_text: $prep_text})-[:HAS_VERDICT]->(v:Verdict)
        WHERE v.verdict_label IN ["True", "False"] AND v.confidence >= $min_confidence
        RETURN c.text AS original_claim,
               v.verdict_label AS verdict_label,
               v.confidence AS confidence,
               v.explanation AS explanation,
               c.timestamp AS timestamp
        ORDER BY v.confidence DESC, c.timestamp DESC // Prioritize higher confidence, then newer
        LIMIT 1
        """
        params = {"prep_text": preprocessed_claim, "min_confidence": KG_CONFIDENCE_THRESHOLD}

        try:
            with self.neo4j_driver.session(database=NEO4J_DATABASE) as session:
                result = session.run(query, params).single() # Fetch at most one record

            if result:
                logging.info(f"KG HIT: Found existing verdict for preprocessed '{preprocessed_claim}' -> '{result['verdict_label']}' (Conf: {result['confidence']:.2f}, Orig: '{result['original_claim'][:50]}...')")
                # Construct the result dictionary to match the output format
                kg_result_dict = {
                    "original_claim": result["original_claim"],
                    "preprocessed_claim": preprocessed_claim, # Include the key used for lookup
                    "ner_entities": [], # NER not stored with verdict, could query if needed
                    "factual_score": None, # Factual score not stored, could be recomputed or stored
                    "initial_verdict_raw": "From KG",
                    "initial_evidence": "From KG",
                    "rag_status": "N/A (From KG)",
                    "top_rag_snippets": [], # Evidence snippets could be retrieved if needed
                    "final_label": result["verdict_label"],
                    "confidence": result["confidence"],
                    "final_explanation": result["explanation"],
                    "source": "Knowledge Graph", # Clearly mark the source
                    "kg_timestamp": result["timestamp"] # Include timestamp of stored claim
                }
                return kg_result_dict
            else:
                logging.info(f"KG MISS: No reliable verdict found for preprocessed '{preprocessed_claim}'.")
                return None
        except neo4j_exceptions.ServiceUnavailable as e:
            logging.error(f"KG Check Failed: Neo4j connection error during query: {e}")
            # Potentially re-disable driver if connection is lost? Or just log.
            return None
        except Exception as e:
            logging.error(f"KG Check Failed: Error querying Neo4j for '{preprocessed_claim}': {e}", exc_info=True)
            return None

    def store_in_neo4j(self, claim_data):
        """Stores the processed claim, verdict, entities, and evidence snippets in Neo4j."""
        if not self.neo4j_driver:
           # logging.error("Neo4j driver not initialized. Cannot store data.") # Logged at init
           return
        if claim_data.get("source") == "Knowledge Graph":
             logging.debug(f"Skipping Neo4j store for claim already retrieved from KG: '{claim_data.get('original_claim', '?')[:50]}...'")
             return

        # Extract data, ensuring basic type safety and defaults
        claim = claim_data.get('original_claim', 'Unknown Claim')
        preprocessed_claim = claim_data.get('preprocessed_claim', '') # Must have preprocessed text
        final_label = claim_data.get('final_label', 'Error')
        confidence = claim_data.get('confidence', 0.1)
        final_explanation = claim_data.get('final_explanation', 'N/A')

        # --- Data Validation & Sanitization ---
        if not preprocessed_claim:
            logging.error(f"Cannot store claim in Neo4j: Missing 'preprocessed_claim' key for '{claim[:50]}...'")
            return

        try:
            confidence = float(confidence)
            if not (0.0 <= confidence <= 1.0):
                 logging.warning(f"Confidence {confidence} out of range for claim '{claim[:50]}...'. Clamping to [0.0, 1.0].")
                 confidence = max(0.0, min(1.0, confidence))
        except (ValueError, TypeError):
            logging.warning(f"Invalid confidence type '{type(confidence)}' value '{confidence}' for claim '{claim[:50]}...'. Using 0.1.")
            confidence = 0.1

        if not isinstance(final_label, str) or final_label not in ["True", "False", "Error", "LLM Error", "Processing Error", "Missing Result", "Preprocessing Error"]:
            logging.warning(f"Invalid final_label '{final_label}' type '{type(final_label)}' for claim '{claim[:50]}...'. Converting to Error string.")
            final_label = "Error" # Store unexpected labels as Error

        if not isinstance(final_explanation, str): final_explanation = str(final_explanation)

        entities = claim_data.get('ner_entities', [])
        if not isinstance(entities, list): entities = []

        initial_evidence = claim_data.get('initial_evidence', "")
        if not isinstance(initial_evidence, str): initial_evidence = str(initial_evidence)

        rag_status = claim_data.get('rag_status', "")
        if not isinstance(rag_status, str): rag_status = str(rag_status)

        initial_verdict_raw = claim_data.get('initial_verdict_raw', "")
        if not isinstance(initial_verdict_raw, str): initial_verdict_raw = str(initial_verdict_raw)

        factual_score = claim_data.get('factual_score', 0.0)
        try: factual_score = float(factual_score) if factual_score is not None else 0.0
        except (ValueError, TypeError): factual_score = 0.0

        top_rag_snippets_raw = claim_data.get('top_rag_snippets', [])
        if not isinstance(top_rag_snippets_raw, list): top_rag_snippets_raw = []
        # Parse snippet strings back into structured data if needed (basic attempt)
        top_rag_snippets_parsed = []
        for item in top_rag_snippets_raw:
            if isinstance(item, dict): # Already structured?
                top_rag_snippets_parsed.append(item)
            elif isinstance(item, str): # Try parsing the formatted string back
                 content = item; source = "?"; index = -1
                 try: # Basic parsing attempt based on format in process_claim
                     if 'Snip ' in item and ': "' in item and '" (' in item:
                          index_part, rest = item.split(': "', 1)
                          index = int(index_part.split(' ')[1]) - 1
                          content_part, source_part = rest.rsplit('" (', 1)
                          content = content_part
                          source = source_part[:-1] if source_part.endswith(')') else source_part
                     top_rag_snippets_parsed.append({'content': content, 'source': source, 'index': index})
                 except Exception:
                      top_rag_snippets_parsed.append({'content': item, 'source': '?', 'index': -1}) # Fallback
            # else: ignore non-string/dict items

        timestamp = time.time() # Current timestamp for update/creation
        # --- End Validation ---


        # --- Neo4j Transaction ---
        # Use a single transaction for atomicity
        with self.neo4j_driver.session(database=NEO4J_DATABASE) as session:
            tx = None
            try:
                tx = session.begin_transaction()

                # MERGE Claim node based on preprocessed_text (our unique key)
                # Update properties on match or set on create
                # Use elementId() to get stable internal ID for relationships
                claim_node_result = tx.run("""
                    MERGE (c:Claim {preprocessed_text: $preprocessed_text})
                    ON CREATE SET
                        c.text = $text,
                        c.timestamp = $timestamp,
                        c.initial_verdict_raw = $initial_verdict_raw,
                        c.rag_status = $rag_status,
                        c.initial_evidence = $initial_evidence,
                        c.factual_score = $factual_score
                    ON MATCH SET
                        c.timestamp = $timestamp,
                        c.initial_verdict_raw = $initial_verdict_raw,
                        c.rag_status = $rag_status,
                        c.initial_evidence = $initial_evidence,
                        c.factual_score = $factual_score,
                        c.text = CASE WHEN c.text <> $text THEN $text ELSE c.text END // Update original text if different
                    RETURN elementId(c) as claim_id
                """, text=claim, preprocessed_text=preprocessed_claim, timestamp=timestamp,
                   initial_verdict_raw=initial_verdict_raw, rag_status=rag_status,
                   initial_evidence=initial_evidence, factual_score=factual_score).single()

                if not claim_node_result: raise Exception(f"Claim node MERGE failed for preprocessed text: {preprocessed_claim}")
                claim_node_id = claim_node_result['claim_id']

                # --- Verdict Handling: Delete old verdict, create new one ---
                # Delete existing HAS_VERDICT relationship and the old Verdict node
                tx.run("MATCH (c:Claim) WHERE elementId(c) = $claim_node_id "
                       "OPTIONAL MATCH (c)-[r:HAS_VERDICT]->(old_v:Verdict) " # Use OPTIONAL MATCH to handle cases where no verdict exists yet
                       "DELETE r, old_v", claim_node_id=claim_node_id)

                # Create the new Verdict node
                verdict_node_result = tx.run("""
                    CREATE (v:Verdict {verdict_label: $verdict_label, confidence: $confidence, explanation: $explanation})
                    RETURN elementId(v) as verdict_id
                    """, verdict_label=final_label, confidence=confidence, explanation=final_explanation).single()
                if not verdict_node_result: raise Exception("Verdict node creation failed.")
                verdict_id = verdict_node_result['verdict_id']

                # Create the new HAS_VERDICT relationship
                tx.run("MATCH (c), (v) WHERE elementId(c) = $claim_node_id AND elementId(v) = $verdict_id "
                       "CREATE (c)-[:HAS_VERDICT]->(v)", claim_node_id=claim_node_id, verdict_id=verdict_id)

                # --- Evidence Snippet Handling: Delete old, create new ---
                # Delete existing HAS_EVIDENCE relationships and old EvidenceSnippet nodes
                tx.run("MATCH (c:Claim) WHERE elementId(c) = $claim_node_id "
                       "OPTIONAL MATCH (c)-[r:HAS_EVIDENCE]->(old_es:EvidenceSnippet) "
                       "DELETE r, old_es", claim_node_id=claim_node_id)

                # Create new EvidenceSnippet nodes and relationships
                for i, snippet_data in enumerate(top_rag_snippets_parsed):
                    content = snippet_data.get('content', '')[:500] # Limit snippet length if needed
                    source = snippet_data.get('source', '?')
                    snippet_index = snippet_data.get('index', i) # Use parsed index or fallback to loop index

                    evidence_snippet_node_result = tx.run("""
                        CREATE (es:EvidenceSnippet {content: $content, source: $source, snippet_index: $snippet_index})
                        RETURN elementId(es) as snippet_id
                        """, content=content, source=source, snippet_index=snippet_index).single()
                    if evidence_snippet_node_result:
                        evidence_snippet_id = evidence_snippet_node_result['snippet_id']
                        # Create relationship from Claim to the new EvidenceSnippet
                        tx.run("MATCH (c), (es) WHERE elementId(c) = $claim_node_id AND elementId(es) = $evidence_snippet_id "
                               "CREATE (c)-[:HAS_EVIDENCE]->(es)", claim_node_id=claim_node_id, evidence_snippet_id=evidence_snippet_id)

                # --- Entity Handling: MERGE entities and MENTIONS relationship ---
                # (Optionally clear old MENTIONS relationships first if entities might change drastically)
                # tx.run("MATCH (c:Claim) WHERE elementId(c) = $claim_node_id OPTIONAL MATCH (c)-[r:MENTIONS]->() DELETE r", claim_node_id=claim_node_id)

                for entity in entities:
                    entity_text = entity.get('text')
                    entity_label = entity.get('label')
                    if not entity_text or not entity_label or entity_label not in KG_RELEVANT_NER_LABELS: # Ensure label is valid
                        continue # Skip invalid entities

                    # MERGE Entity node (create if not exists)
                    entity_node_result = tx.run(
                        "MERGE (e:Entity {text: $text, label: $label}) RETURN elementId(e) as entity_id",
                        text=entity_text, label=entity_label
                    ).single()

                    if entity_node_result:
                        entity_id = entity_node_result['entity_id']
                        # MERGE relationship: Create MENTIONS if it doesn't exist between this claim and entity
                        tx.run("MATCH (c), (e) WHERE elementId(c) = $claim_node_id AND elementId(e) = $entity_id "
                               "MERGE (c)-[:MENTIONS]->(e)", claim_node_id=claim_node_id, entity_id=entity_id)

                # Commit the transaction if all steps succeeded
                tx.commit()
                logging.info(f"Stored/Updated claim '{claim[:50]}...' in Neo4j (DB: {NEO4J_DATABASE}).")

            except Exception as e:
                if tx and not tx.closed(): # Check if tx exists and is not already closed (e.g., by commit/rollback)
                    logging.error(f"Transaction failed for claim '{claim[:50]}...'. Rolling back Neo4j changes.")
                    try: tx.rollback()
                    except Exception as rb_e: logging.error(f"Error during Neo4j rollback: {rb_e}")
                logging.error(f"Error storing/updating claim '{claim[:50]}...' in Neo4j: {e}", exc_info=True)
        # --- End Neo4j Transaction ---

    def preprocess_and_filter(self, text: str) -> (list, list):
        """Segments text into sentences, filters non-checkable ones, and preprocesses checkable ones for KG lookup."""
        if not text or not isinstance(text, str):
            logging.warning("Preprocess & Filter: Input text is empty or invalid."); return [], []

        sentences = []; checkable_claims_data = []; non_checkable_claims = []
        try:
            # Basic sentence splitting improvements
            text = text.replace('.This','. This').replace('?This','? This').replace('!This','! This');
            sentences = nltk.sent_tokenize(text);
            sentences = [s.strip() for s in sentences if len(s.strip()) > 1]; # Remove empty/whitespace-only sentences
            logging.info(f"Segmented into {len(sentences)} potential sentences.")
        except Exception as e:
            logging.error(f"NLTK sentence tokenization failed: {e}", exc_info=True);
            # Fallback: split by newline or just use the whole text?
            sentences = [text.strip()] if text.strip() else []
            if not sentences: return [], [] # Cannot proceed if tokenization fails badly

        for sentence in sentences:
            if is_claim_checkable(sentence):
                sentence_ner = [];
                # Extract NER entities if NLP is available
                if self.nlp_available:
                    try:
                        doc = NLP(sentence)
                        # Filter entities by relevance
                        sentence_ner = [{"text": ent.text, "label": ent.label_} for ent in doc.ents if ent.label_ in KG_RELEVANT_NER_LABELS]
                        logging.debug(f"NER for checkable claim '{sentence[:50]}...': {sentence_ner}")
                    except Exception as e:
                        logging.error(f"NER extraction failed for sentence '{sentence[:50]}...': {e}")
                        sentence_ner = [] # Ensure it's an empty list on error

                # Generate the preprocessed version specifically for KG lookup
                preprocessed_for_kg = preprocess_claim_for_kg(sentence)

                # Ensure preprocessed text for KG is not empty (should fallback to original)
                if not preprocessed_for_kg:
                     logging.warning(f"KG Preprocessing resulted in empty string for '{sentence}'. Skipping checkable status.")
                     non_checkable_claims.append(sentence + " [KG Preprocessing Failed]")
                     continue # Treat as non-checkable if KG key generation fails

                checkable_claims_data.append({
                    "original_claim": sentence,
                    "preprocessed_claim": preprocessed_for_kg, # Key for KG matching
                    "ner_entities": sentence_ner # Store extracted entities
                })
            else:
                # Sentence was filtered by is_claim_checkable
                non_checkable_claims.append(sentence)

        logging.info(f"Preprocessing complete. Found {len(checkable_claims_data)} checkable claims. Filtered {len(non_checkable_claims)} non-checkable sentences.")
        return checkable_claims_data, non_checkable_claims

    def classify_and_prioritize_claims(self, checkable_claims_data: list) -> list:
        """Assigns a 'factual score' and priority to checkable claims based on embedding norms."""
        # No changes needed here based on requirements - uses original_claim for scoring
        if not checkable_claims_data: logging.warning("Prioritize: No data provided."); return []
        try:
            # Filter out any entries that might lack 'original_claim' before encoding
            valid_claims_data = [cd for cd in checkable_claims_data if cd.get('original_claim') and isinstance(cd.get('original_claim'), str)]
            if not valid_claims_data:
                logging.warning("Prioritize: No valid claims with text found.")
                return []

            original_claims_texts = [cd["original_claim"] for cd in valid_claims_data]

            # Generate embeddings for the original claim text
            embeddings = self.embedding_model.encode(original_claims_texts, convert_to_tensor=True, show_progress_bar=False)

            # Calculate scores and priorities
            for i, claim_data in enumerate(valid_claims_data):
                embedding_vector = embeddings[i].cpu().numpy()
                norm = np.linalg.norm(embedding_vector)
                # Heuristic: higher norm might correlate with more specific/factual content?
                # Normalize score to be roughly between 0.5 and 1.0
                score = 0.5 + (norm / (norm * 25 + 1e-6)) if norm > 1e-6 else 0.5
                score = min(max(score, 0.0), 1.0) # Clamp score to [0.0, 1.0]
                priority = score # Use the score directly as priority for now
                claim_data["factual_score"] = score
                claim_data["priority"] = priority
                logging.debug(f"Prioritize: '{claim_data['original_claim'][:50]}...' -> Score:{score:.3f}, Prio:{priority:.3f}")

            # Sort claims by priority (highest first)
            valid_claims_data.sort(key=lambda x: x.get('priority', 0.0), reverse=True)
            logging.info(f"Prioritized {len(valid_claims_data)} claims based on factual score heuristic.")
            return valid_claims_data

        except Exception as e:
            logging.error(f"Claim prioritization failed: {e}", exc_info=True)
            return checkable_claims_data # Return original list on error to avoid losing data

    def add_claims_to_queue(self, claims_to_process: list):
        """Adds claims needing full processing to the worker queue."""
        # No changes needed here
        if not claims_to_process:
            logging.warning("Queue: No claims provided to add."); return
        for claim_data_dict in claims_to_process:
            if isinstance(claim_data_dict, dict) and claim_data_dict.get('original_claim'):
                self.claim_queue.put(claim_data_dict)
            else:
                logging.warning(f"Queue: Skipping invalid claim data item: {claim_data_dict}")
        logging.info(f"Queued {len(claims_to_process)} claims for full processing. Current Queue Size: {self.claim_queue.qsize()}")

    # --- MODIFIED process_claim (v2) ---
    def process_claim(self, claim_data_dict: dict):
        """
        Processes a single claim that was *not* found in the KG.
        Uses ORIGINAL claim for API calls (GFactCheck, GSearch).
        Uses ORIGINAL claim for RAG similarity search.
        Uses ORIGINAL claim for LLM synthesis (which now includes internal knowledge).
        Stores result in Neo4j using PREPROCESSED claim as key.
        """
        original_claim = claim_data_dict.get('original_claim')
        preprocessed_claim = claim_data_dict.get('preprocessed_claim') # Needed for KG storage key
        ner_entities = claim_data_dict.get('ner_entities', [])
        factual_score = claim_data_dict.get('factual_score', 0.0)

        if not original_claim or not preprocessed_claim:
            logging.error(f"Process Claim Error: Missing original or preprocessed claim text in dict: {claim_data_dict}")
            # Store error result?
            with self.results_lock:
                 self.results[original_claim or f"Unknown_{time.time()}"] = {
                     "original_claim": original_claim or 'Unknown', "preprocessed_claim": preprocessed_claim or 'Unknown',
                     "ner_entities": ner_entities, "factual_score": factual_score,
                     "final_label": "Processing Error", "confidence": 0.1,
                     "final_explanation": "Input data missing original or preprocessed claim.", "source": "Error"
                 }
            return

        start_time = time.time()
        logging.info(f"Full Process Start: \"{original_claim[:60]}...\"")

        # Initialize result dictionary
        result = {
            "original_claim": original_claim, "preprocessed_claim": preprocessed_claim, "ner_entities": ner_entities,
            "factual_score": factual_score,
            "initial_verdict_raw": "N/A", "initial_evidence": "N/A", # Step 1 result
            "rag_status": "Not Attempted", "top_rag_snippets": [], # Step 2/3 results
            "final_label": "Pending", "confidence": 0.0, "final_explanation": "N/A", # Final results from synthesis
            "source": "Full Pipeline" # Mark as processed by this path
        }
        rag_evidence_for_llm = [] # Store structured RAG results for LLM

        # --- Step 1: Initial Fact Check (Google Fact Check API - using ORIGINAL claim) ---
        initial_check_verdict = "N/A"
        initial_check_evidence = "N/A"
        try:
            # Use ORIGINAL claim for the API query
            initial_check_list = google_fact_check(original_claim, GOOGLE_API_KEY)
            self.raw_fact_checks[original_claim] = initial_check_list # Store raw response
            if initial_check_list:
                 initial_check = initial_check_list[0] # Take the first result
                 initial_check_verdict = initial_check.get('verdict', 'Error')
                 initial_check_evidence = initial_check.get('evidence', 'N/A')
                 result['initial_verdict_raw'] = initial_check_verdict
                 result['initial_evidence'] = initial_check_evidence
                 # Handle potential errors reported by the API itself
                 if initial_check_verdict == 'Error':
                     logging.warning(f"GFactCheck API returned error for '{original_claim[:50]}...': {initial_check_evidence}")
            else:
                 # API returned empty list (but no exception)
                 result['initial_verdict_raw'] = 'Error'; result['initial_evidence'] = 'API returned no data'
                 initial_check_verdict = 'Error' # Ensure these are set for LLM input if API fails silently
                 initial_check_evidence = 'API returned no data'
            logging.info(f"GFactCheck Result for '{original_claim[:50]}...': '{initial_check_verdict}'")
        except Exception as e: # Catch unexpected errors in this block
            logging.error(f"GFactCheck call failed unexpectedly for '{original_claim[:50]}...': {e}", exc_info=True)
            result['initial_verdict_raw']="Error"; result['initial_evidence']=f"GFactCheck Exception: {e}"
            initial_check_verdict = 'Error'; initial_check_evidence = f"GFactCheck Exception: {e}"

        # --- Step 2: Google Custom Search (For RAG - using ORIGINAL claim) ---
        search_results = []; full_search_resp = {}
        try:
            logging.debug(f"Attempting GCustomSearch for '{original_claim[:60]}...'");
            # Use ORIGINAL claim for the search query
            full_search_resp, search_results = google_custom_search(original_claim, GOOGLE_API_KEY, GOOGLE_CSE_ID, NUM_SEARCH_RESULTS)
            # Store raw response and parsed results
            self.raw_searches[original_claim] = {"query": original_claim, "response": full_search_resp, "results": search_results}

            # Update RAG status based on search outcome
            if not search_results and full_search_resp: # API OK, but no items found
                result["rag_status"] = "Search OK, No Results"
            elif not search_results and not full_search_resp: # API call likely failed (e.g., 429, network error)
                result["rag_status"] = "Search Failed (API Error)"
            elif search_results: # Got results
                result["rag_status"] = "Search OK, Results Found"
            else: # Should not happen if full_search_resp is None/empty and search_results is empty, but catch anyway
                 result["rag_status"] = "Search Failed (Unknown Error)"
            logging.info(f"GCustomSearch status for '{original_claim[:50]}...': {result['rag_status']}")
        except Exception as e:
            logging.error(f"GCustomSearch call failed unexpectedly for '{original_claim[:50]}...': {e}", exc_info=True)
            result["rag_status"]="Search Failed (Exception)"
            search_results=[] # Ensure search_results is empty list on error

        # --- Step 3: RAG (Vector Search on Search Results - using ORIGINAL claim for retrieval) ---
        if search_results:
            documents=[]; vector_store=None
            try:
                min_snippet_len = 20 # Minimum characters for a snippet to be considered useful
                # Filter results based on minimum length
                valid_search_results = [sr for sr in search_results if len(sr.get('snippet', '')) >= min_snippet_len]

                if not valid_search_results:
                     result["rag_status"] = "Search OK, No Usable Snippets (Too Short)"
                     logging.info(f"RAG Step skipped for '{original_claim[:50]}...': No search snippets met minimum length {min_snippet_len}.")
                else:
                    # Optional: Filter snippets for relevance to the original claim keywords
                    # This might be redundant if similarity search works well, but can prune irrelevant results early
                    claim_words = {w.lower() for w in original_claim.translate(str.maketrans('', '', string.punctuation)).split() if len(w)>2}
                    relevant_search_results = [sr for sr in valid_search_results if any(w in sr['snippet'].lower() for w in claim_words)]

                    if not relevant_search_results:
                         result["rag_status"] = "Search OK, No Relevant Snippets Found"
                         logging.info(f"RAG Step skipped for '{original_claim[:50]}...': No snippets seemed relevant based on keyword check.")
                    else:
                        # Create LangChain Documents
                        documents = [Document(page_content=sr['snippet'], metadata={'source':sr['link'],'title':sr['title']})
                                     for sr in relevant_search_results]

                        if documents:
                            try:
                                # Create FAISS index from the relevant documents
                                vector_store=FAISS.from_documents(documents, self.langchain_embeddings);
                                logging.debug(f"FAISS index created from {len(documents)} documents for '{original_claim[:50]}...'.")

                                # Perform similarity search using the ORIGINAL claim
                                retrieved_docs = vector_store.similarity_search(original_claim, k=RAG_K);

                                # Store structured results for LLM and formatted strings for output
                                rag_evidence_for_llm = [{"content":doc.page_content,"metadata":doc.metadata} for doc in retrieved_docs]
                                result['top_rag_snippets'] = [f"Snip {j+1}: \"{d['content'][:150].strip()}...\" ({d['metadata'].get('source','?')})" for j,d in enumerate(rag_evidence_for_llm)]
                                result["rag_status"] = f"RAG OK ({len(rag_evidence_for_llm)}/{len(documents)} snippets retrieved/indexed)"
                                logging.info(f"RAG status for '{original_claim[:50]}...': {result['rag_status']}")

                            except Exception as faiss_e:
                                 logging.error(f"FAISS/Embedding error during RAG for '{original_claim[:50]}...': {faiss_e}", exc_info=True)
                                 result["rag_status"] = f"RAG Failed (FAISS/Embedding Error)"
                                 rag_evidence_for_llm = [] # Clear evidence on error
                                 result['top_rag_snippets'] = []
                        else:
                            # This case should be caught by `if not relevant_search_results` check above, but handle defensively
                            result["rag_status"] = "RAG Failed (No Documents after filtering)"
                            rag_evidence_for_llm = []
                            result['top_rag_snippets']=[]

            except Exception as e: # Catch errors in the outer RAG try block (e.g., keyword processing)
                logging.error(f"Outer RAG processing failed for '{original_claim[:50]}...': {e}",exc_info=True);
                result["rag_status"]=f"RAG Failed (Processing Error: {type(e).__name__})"
                rag_evidence_for_llm = [] # Clear evidence on error
                result['top_rag_snippets'] = []
        else:
             # RAG status was already set based on search results (e.g., "Search Failed", "No Results")
             # Ensure evidence lists are empty if search failed or yielded no results
             rag_evidence_for_llm = []
             result['top_rag_snippets'] = []
             logging.info(f"RAG Step skipped for '{original_claim[:50]}...' due to previous search status: {result['rag_status']}")

        # --- Step 4: LLM Final Verdict (Synthesizing Initial Check + RAG + Internal Knowledge) ---
        logging.info(f"LLM generating final synthesized verdict for '{original_claim[:50]}...' (using internal knowledge)...");
        # Use the revised LLM function v2
        llm_final_result = get_llm_final_verdict_v2(
            claim=original_claim, # Pass original claim
            initial_check_verdict=initial_check_verdict,
            initial_check_evidence=initial_check_evidence,
            rag_evidence=rag_evidence_for_llm, # Pass structured RAG results
            rag_status_msg=result['rag_status']
        )
        # Store the synthesized results from the LLM
        result['final_label'] = llm_final_result['final_label']
        result['confidence'] = llm_final_result['confidence']
        result['final_explanation'] = llm_final_result['explanation']

        # --- Step 5: Store in Neo4j (using preprocessed_claim as key) ---
        try:
            # Pass the final consolidated 'result' dictionary which includes the preprocessed_claim
            self.store_in_neo4j(result)
        except Exception as neo4j_e:
            # Log error but don't overwrite the final verdict if storage fails
            logging.error(f"Neo4j storage failed for claim '{original_claim[:50]}...': {neo4j_e}", exc_info=True)

        # --- Final Logging ---
        processing_time = time.time() - start_time
        logging.info(
            f"Final Verdict (Synthesized v2) for '{original_claim[:50]}...': "
            f"{result['final_label']} (Conf: {result['confidence']:.2f}). "
            f"(Explain: {result['final_explanation'][:80]}...). "
            f"(Time:{processing_time:.2f}s)"
        )
        # Store the final result (including LLM synthesis) in the shared dictionary
        with self.results_lock:
            self.results[original_claim] = result

    # --- Worker Function (unchanged) ---
    def worker(self):
        """Worker thread function to process claims from the queue."""
        t_obj = current_thread(); t_name = t_obj.name; logging.info(f"Worker {t_name} started.")
        while True:
            claim_data_dict = None # Keep track of item being processed for error handling
            try:
                # Get a claim dictionary from the queue, wait up to 1 second if empty
                claim_data_dict = self.claim_queue.get(timeout=1)
                if claim_data_dict is None: # Use None as a sentinel value to signal workers to exit
                    break # Exit loop if sentinel received

                # Process the claim using the main processing logic
                self.process_claim(claim_data_dict)

                # Mark the task as done in the queue
                self.claim_queue.task_done()

            except queue.Empty:
                # Queue was empty for the timeout duration, worker can finish
                logging.info(f"Worker {t_name} found queue empty, finishing.")
                break # Exit loop

            except Exception as e:
                # Handle unexpected errors during claim processing
                original_claim = claim_data_dict.get('original_claim', '?') if claim_data_dict else '?'
                logging.error(f"Worker {t_name} encountered an error processing claim '{original_claim[:50]}...': {e}", exc_info=True)

                # Store an error result for this claim if possible
                if claim_data_dict and original_claim != '?':
                    with self.results_lock:
                        self.results[original_claim] = {
                            "original_claim": original_claim,
                            "preprocessed_claim": claim_data_dict.get('preprocessed_claim','?'),
                            "ner_entities": claim_data_dict.get('ner_entities', []),
                            "factual_score": claim_data_dict.get('factual_score', 0.0),
                            "final_label": "Processing Error",
                            "confidence": 0.1,
                            "final_explanation": f"Worker thread encountered error: {str(e)}",
                            "source": "Error"
                            }
                # Mark task as done even if it failed, to prevent queue blockage
                try: self.claim_queue.task_done()
                except ValueError: pass # task_done might raise error if called too many times

        logging.info(f"Worker {t_name} finished.")

    # --- SHAP Function (unchanged, but acknowledges issues) ---
    def train_and_run_shap(self, claims_processed_fully: list):
        """Attempts SHAP analysis on fully processed claims. Uses original claim text."""
        # SHAP requires the 'shap' library
        if not self.shap_available:
            logging.warning("SHAP unavailable (library not installed). Skipping SHAP analysis.")
            # Create placeholder results if SHAP is unavailable
            self.shap_explanations = [{"claim": cd.get('original_claim', '?'), "shap_values": "[SHAP Unavailable]"}
                                      for cd in claims_processed_fully if cd.get('original_claim')]
            return

        if not claims_processed_fully:
            logging.info("SHAP: No claims went through the full pipeline, skipping SHAP analysis.")
            self.shap_explanations = []
            return

        # Filter for valid data: must have original_claim as a string
        valid_claims_data = [cd for cd in claims_processed_fully if cd.get("original_claim") and isinstance(cd.get("original_claim"), str)]
        if not valid_claims_data:
            logging.warning("SHAP: No valid claims data with text found for SHAP analysis.")
            self.shap_explanations = []
            return

        logging.info(f"Attempting SHAP explanations for {len(valid_claims_data)} fully processed claims...");
        sentences = [cd['original_claim'] for cd in valid_claims_data]

        # Initialize explanations with pending status
        self.shap_explanations = [{"claim": s, "shap_values": "[SHAP Pending]"} for s in sentences]

        try:
            # Get embedding dimension dynamically if possible
            embed_dim = self.embedding_model.get_sentence_embedding_dimension()
            logging.debug(f"SHAP using embedding dimension: {embed_dim}")
        except Exception as e:
            logging.warning(f"Could not get embedding dimension dynamically for SHAP: {e}. Using fallback 384.")
            embed_dim = 384 # Fallback dimension for MiniLM-L6-v2

        try:
            # 1. Get Embeddings (use numpy arrays for SHAP)
            embeddings = self.embedding_model.encode(sentences, convert_to_tensor=False, show_progress_bar=False);
            if not isinstance(embeddings, np.ndarray) or embeddings.ndim != 2 or embeddings.shape[0] != len(sentences) or embeddings.shape[1] != embed_dim:
                 logging.error(f"SHAP: Invalid embeddings generated. Type: {type(embeddings)}, Shape: {getattr(embeddings, 'shape', 'N/A')}. Expected: ({len(sentences)}, {embed_dim}).")
                 self.shap_explanations = [{"claim": s, "shap_values": "[SHAP Embed Err]"} for s in sentences]; return
            logging.debug(f"SHAP embeddings generated, shape: {embeddings.shape}")

            # 2. Define Prediction Function for SHAP
            #    This function takes a NumPy array of embeddings and returns a NumPy array of scores.
            #    Currently uses the 'factual_score' heuristic based on norm.
            #    *** THIS IS LIKELY THE CAUSE OF ZERO SHAP VALUES ***
            #    The score function is simple and may not have enough variance or gradient
            #    for SHAP's perturbation method to assign importance. A more complex model
            #    trained on the embeddings might be needed for meaningful SHAP values.
            def predict_scores_for_shap(np_embeds):
                 # Input validation
                 if not isinstance(np_embeds, np.ndarray):
                     try: # Attempt conversion if list-like structure passed
                         np_embeds = np.array(np_embeds)
                         if np_embeds.ndim == 1: np_embeds = np_embeds.reshape(1, -1) # Reshape 1D array
                         assert np_embeds.ndim == 2 # Ensure 2D array
                     except Exception as pred_e:
                          logging.error(f"SHAP predict input error: {pred_e}. Input type: {type(np_embeds)}. Returning default 0.5.")
                          num_inputs = len(np_embeds) if hasattr(np_embeds, '__len__') else 1
                          return np.full(num_inputs, 0.5) # Return neutral score on error

                 # Calculate scores based on embedding norm heuristic
                 scores = []
                 for e in np_embeds:
                     e_1d = e.flatten()
                     norm = float(np.linalg.norm(e_1d))
                     score = 0.5 + (norm / (norm * 25 + 1e-6)) if norm > 1e-6 else 0.5
                     scores.append(min(max(score, 0.0), 1.0)) # Clamp score [0.0, 1.0]
                 return np.array(scores)

            # 3. Create Background Dataset for SHAP Explainer
            logging.debug("Creating SHAP background data..."); bg_data = None
            n_samples = embeddings.shape[0]
            n_bg_samples = min(100, n_samples) # Limit background samples
            n_clusters = min(10, n_bg_samples) # Number of clusters for KMeans

            try:
                 # Use KMeans for a representative background set if enough samples/clusters
                 if n_samples >= n_clusters and n_clusters > 1:
                     bg_obj = shap.kmeans(embeddings, n_clusters)
                     # Check if kmeans output is as expected
                     if hasattr(bg_obj, 'data') and isinstance(bg_obj.data, np.ndarray):
                         bg_data = bg_obj.data
                         logging.debug(f"Using KMeans background data, shape: {bg_data.shape}")
                     else: # Fallback if kmeans fails or returns unexpected format
                         logging.warning(f"SHAP KMeans object missing '.data' attribute or not ndarray. Using random subset instead. Kmeans result: {bg_obj}")
                         indices = np.random.choice(n_samples, n_bg_samples, replace=False)
                         bg_data = embeddings[indices]
                 # If not enough samples for clustering, use a random subset
                 elif n_samples > 0:
                     logging.debug(f"Using random subset of embeddings for background (samples: {n_samples}, subset size: {n_bg_samples})")
                     indices = np.random.choice(n_samples, n_bg_samples, replace=False)
                     bg_data = embeddings[indices]
                 else: # No embeddings to create background from
                     logging.error("SHAP Error: Cannot create background data - No embeddings available.")
                     self.shap_explanations = [{"claim": s, "shap_values": "[SHAP BG Err]"} for s in sentences]; return
            except ImportError as imp_err: # Catch missing dependency like sklearn for KMeans
                 logging.warning(f"SHAP KMeans dependency possibly missing ({imp_err}). Using random subset for background data.")
                 indices = np.random.choice(n_samples, n_bg_samples, replace=False)
                 bg_data = embeddings[indices]
            except Exception as ke: # Catch other KMeans errors
                 logging.warning(f"SHAP KMeans failed: {ke}. Using random subset for background data.")
                 indices = np.random.choice(n_samples, n_bg_samples, replace=False)
                 bg_data = embeddings[indices]

            # Final check on background data
            if bg_data is None or not isinstance(bg_data, np.ndarray) or bg_data.shape[0] == 0:
                logging.error("SHAP background data preparation failed or resulted in empty data.");
                self.shap_explanations = [{"claim": s, "shap_values": "[SHAP BG Err]"} for s in sentences]; return

            # Ensure background data is 2D
            if bg_data.ndim == 1: bg_data = bg_data.reshape(1, -1)
            logging.debug(f"Final background data shape for SHAP: {bg_data.shape}")

            # 4. Initialize SHAP Explainer
            logging.debug("Initializing SHAP KernelExplainer...");
            # KernelExplainer works by perturbing features (embeddings dimensions here)
            explainer = shap.KernelExplainer(predict_scores_for_shap, bg_data)

            # 5. Calculate SHAP Values
            logging.info(f"Calculating SHAP values for {embeddings.shape[0]} instance(s)...");
            # nsamples='auto' or a reasonable number based on features
            # Link='identity' since we are predicting scores directly
            n_samples_shap = min(50, 2 * embeddings.shape[1] + 2048) # SHAP default heuristic
            shap_vals = explainer.shap_values(embeddings, nsamples=n_samples_shap, l1_reg='auto') # Added l1_reg='auto'
            logging.info("SHAP values calculation completed.")

            # 6. Store SHAP Values
            calculated_explanations = []
            if isinstance(shap_vals, np.ndarray):
                expected_shape_single = (embed_dim,) # Shape if only one sentence
                expected_shape_multi = (len(sentences), embed_dim) # Shape if multiple sentences

                if shap_vals.shape == expected_shape_multi:
                    # Correct shape for multiple sentences
                    for i, sentence in enumerate(sentences):
                        calculated_explanations.append({"claim": sentence, "shap_values": shap_vals[i].tolist()})
                elif len(sentences) == 1 and shap_vals.shape == expected_shape_single:
                    # Correct shape if only one sentence was processed
                     calculated_explanations.append({"claim": sentences[0], "shap_values": shap_vals.tolist()})
                else:
                     # Shape mismatch
                     logging.error(f"SHAP values shape mismatch. Got {shap_vals.shape}, expected {expected_shape_multi} or {expected_shape_single} for 1 sentence.");
                     calculated_explanations = [{"claim": s, "shap_values": f"[SHAP Shape Err: Got {shap_vals.shape}]"} for s in sentences]

            elif isinstance(shap_vals, list) and len(shap_vals) == len(sentences):
                 # Check if list contains numpy arrays of the correct shape
                 if all(isinstance(item, np.ndarray) and item.shape == (embed_dim,) for item in shap_vals):
                      for i, sentence in enumerate(sentences):
                          calculated_explanations.append({"claim": sentence, "shap_values": shap_vals[i].tolist()})
                 else:
                      logging.error(f"SHAP values list content mismatch. Items shapes inconsistent or not ({embed_dim},).");
                      calculated_explanations = [{"claim": s, "shap_values": "[SHAP List Content Err]"} for s in sentences]
            else: # Unexpected type
                logging.error(f"SHAP values returned unexpected type: {type(shap_vals)}. Expected ndarray or list of ndarrays.");
                calculated_explanations = [{"claim": s, "shap_values": f"[SHAP Type Err: {type(shap_vals).__name__}]"} for s in sentences]

            # Log success/failure and store results
            if calculated_explanations:
                all_zero = True
                for expl in calculated_explanations:
                     vals = expl.get("shap_values")
                     if isinstance(vals, list) and any(abs(float(v)) > 1e-9 for v in vals if isinstance(v, (int, float))):
                          all_zero = False; break
                if all_zero and calculated_explanations:
                    logging.warning("SHAP Analysis completed, but all calculated SHAP values are zero or near-zero. The prediction function might be too simple or lack sensitivity for SHAP.")
                else:
                     logging.info(f"SHAP results stored for {len(calculated_explanations)} claims.")
            self.shap_explanations = calculated_explanations

        except Exception as e:
            logging.error(f"SHAP generation failed with unexpected error: {e}", exc_info=True);
            # Store error message in explanations
            self.shap_explanations = [{"claim": s, "shap_values": f"[SHAP Error: {type(e).__name__}]"} for s in sentences]

    # --- Chain of Thought and Check methods (minor adjustments for clarity) ---
    def generate_chain_of_thought(self, all_processed_claims: list, non_checkable_claims: list) -> str:
        """Generates a step-by-step summary of the fact-checking process."""
        cot = ["Chain of Thought Summary:"]

        # Step 1: Input and Filtering
        cot.append("1. Input Segmentation & Filtering:")
        total_initial = len(all_processed_claims) + len(non_checkable_claims)
        cot.append(f"   - Initial Sentences Extracted: {total_initial}")
        if non_checkable_claims:
            # Sort for consistent output order
            non_checkable_sorted = sorted(non_checkable_claims)
            cot.append(f"   - Filtered Non-Checkable ({len(non_checkable_sorted)}): {non_checkable_sorted}")
        cot.append(f"   - Checkable Claims Identified: {len(all_processed_claims)}")

        # Step 2: Preprocessing & Prioritization (for checkable claims)
        # Separate KG hits from those needing full processing before showing prioritization
        kg_hits = [c for c in all_processed_claims if c.get('source') == 'Knowledge Graph']
        pipeline_processed = [c for c in all_processed_claims if c.get('source') == 'Full Pipeline']
        errors_or_preprocessing_failed = [c for c in all_processed_claims if c.get('source') == 'Error' or 'Error' in c.get('final_label','')]

        prioritized_for_pipeline = [c for c in pipeline_processed if 'priority' in c]
        if prioritized_for_pipeline:
            # Sort by original claim text for consistent display
            checkable_claims_str = [f"'{c.get('original_claim','?')[:50]}...' (Prio:{c.get('priority',0.0):.3f})"
                                    for c in sorted(prioritized_for_pipeline, key=lambda x: x.get('original_claim', ''))]
            cot.append(f"   - Prioritized for Full Pipeline ({len(checkable_claims_str)}): [{', '.join(checkable_claims_str)}]")
        elif any('priority' in c for c in all_processed_claims):
             cot.append("   - Prioritization: Attempted, but no claims proceeded to full pipeline.")

        # Step 3: KG Check & Pipeline Execution
        cot.append("2. Knowledge Graph Check & Processing Pipeline:")
        if kg_hits: cot.append(f"   - KG Hits ({len(kg_hits)}): Found existing reliable verdicts, skipped full pipeline.")
        else: cot.append("   - KG Hits: None found meeting criteria.")

        if pipeline_processed: cot.append(f"   - Full Pipeline Execution ({len(pipeline_processed)}): Claims processed via APIs and Synthesizing LLM.")
        else: cot.append("   - Full Pipeline Execution: No claims required or completed full processing.")

        if errors_or_preprocessing_failed:
            error_claims_str = [f"'{c.get('original_claim','?')[:50]}...' ({c.get('final_label', 'Error')})"
                               for c in sorted(errors_or_preprocessing_failed, key=lambda x: x.get('original_claim', ''))]
            cot.append(f"   - Errors ({len(errors_or_preprocessing_failed)}): Claims encountered errors during processing: [{', '.join(error_claims_str)}]")

        # Step 4: Results Summary
        cot.append("3. Processed Claim Results (Sorted by Original Claim):")
        # Sort all results together for final output
        all_processed_claims_sorted = sorted(all_processed_claims, key=lambda x: x.get('original_claim', ''))
        for i, res in enumerate(all_processed_claims_sorted):
                claim = res.get('original_claim','?')
                source = res.get('source', '?')
                final_label = res.get('final_label', '?')
                confidence = res.get('confidence', 0.0)
                explanation = res.get('final_explanation', '?')

                cot.append(f"   - Claim {i+1}: '{claim}'")
                cot.append(f"     - Source: {source}")
                if source == "Knowledge Graph":
                    cot.append(f"     - Verdict: {final_label} (Confidence: {confidence:.2f})")
                    cot.append(f"     - KG Explanation: {explanation}")
                    kg_time = res.get("kg_timestamp")
                    if kg_time: cot.append(f"     - KG Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime(kg_time))}")
                elif source == "Full Pipeline":
                     cot.append(f"     - Initial API Check: Verdict='{res.get('initial_verdict_raw','?')}' (Source: {res.get('initial_evidence','?')})")
                     cot.append(f"     - RAG Status: {res.get('rag_status','?')}")
                     # cot.append(f"     - Top RAG Snippets: {res.get('top_rag_snippets',[])}") # Maybe too verbose for CoT
                     cot.append(f"     - Final Verdict (LLM Synthesis): {final_label} (Confidence: {confidence:.2f})")
                     cot.append(f"     - LLM Justification: {explanation}")
                     if self.neo4j_driver: cot.append(f"     - Neo4j Storage: Stored/Updated")
                elif source == "Error":
                     cot.append(f"     - Verdict: {final_label} (Confidence: {confidence:.2f})")
                     cot.append(f"     - Error Explanation: {explanation}")
                else:
                     cot.append(f"     - Status: Result Missing or Unknown Source!")

        # Step 5: SHAP Summary
        cot.append("4. SHAP Analysis Summary (for fully processed claims):")
        if self.shap_explanations:
             # Filter SHAP results to only include claims that were actually processed by the pipeline
             processed_claims_originals = {c['original_claim'] for c in pipeline_processed}
             relevant_explanations = [ex for ex in self.shap_explanations if ex.get('claim') in processed_claims_originals]

             if relevant_explanations:
                 sh_sum=[]; has_real_values=False; has_errors=False; all_zero=True
                 # Sort by claim for consistent output
                 relevant_explanations_sorted = sorted(relevant_explanations, key=lambda x: x.get('claim', ''))
                 for ex in relevant_explanations_sorted:
                     claim_text = ex.get('claim', '?')[:40] + '...'
                     v = ex.get('shap_values', [])
                     s = "[Status Unknown]" # Default status string

                     if isinstance(v, list) and v:
                         # Check for error strings first
                         if any(isinstance(val, str) and val.startswith('[SHAP') for val in v):
                             s = v[0] if v and isinstance(v[0], str) else "[SHAP Err Fallback]"
                             has_errors = True
                             all_zero = False
                         # Check if all are numeric and near-zero
                         elif all(isinstance(x, (int, float)) and abs(float(x)) < 1e-9 for x in v):
                             s = "[Zero Values]"
                         # Check if numeric and has non-zero values
                         elif all(isinstance(x, (int, float)) for x in v):
                             s = f"[...{len(v)} SHAP vals...]"
                             has_real_values = True
                             all_zero = False
                         else: # Mixed data types or other issue
                             s = "[Mixed/Invalid Data]"
                             all_zero = False
                     elif isinstance(v, str) and v.startswith('[SHAP'): # SHAP failed entirely for this claim
                         s = v
                         has_errors = True
                         all_zero = False
                     elif not v: # Empty list
                         s = "[No SHAP Data]"
                         all_zero = False

                     sh_sum.append(f"'{claim_text}': {s}")

                 # Determine overall SHAP status message
                 status = "Unavailable/Not Run."
                 if has_real_values: status = "Generated values."
                 elif has_errors: status = "Failed/Unavailable (Errors encountered)."
                 elif all_zero and not has_errors and relevant_explanations: status = "Zero values reported (Prediction function likely unsuitable)."
                 elif relevant_explanations: status = "Completed (No non-zero values or errors detected)." # Catch-all if logic above missed something

                 cot.append(f"   - SHAP Status: {status}")
                 if sh_sum: cot.append(f"   - Details: {{{', '.join(sh_sum)}}}")

             elif self.shap_available:
                 cot.append("   - SHAP analysis skipped: No claims processed via full pipeline or SHAP failed before value calculation.")
             else: # SHAP library not installed
                 cot.append("   - SHAP analysis skipped (library not installed).")
        else:
             # self.shap_explanations structure itself is missing or empty
            cot.append("   - SHAP analysis skipped (no results structure generated).")

        return "\n".join(cot)

    def check(self, text: str, num_workers: int = 2) -> dict:
        """
        Main entry point to fact-check a block of text.
        Coordinates preprocessing, KG check, parallel processing, SHAP, and result consolidation.
        """
        start_time = time.time()
        logging.info(f"Starting comprehensive check for text: \"{text[:100]}...\"")

        # --- 0. Reset state for this run ---
        with self.results_lock:
            self.results = {} # Clear previous results
            self.raw_fact_checks = {}
            self.raw_searches = {}
        self.shap_explanations = [] # Clear previous SHAP explanations
        # Clear the worker queue
        while not self.claim_queue.empty():
            try: self.claim_queue.get_nowait(); self.claim_queue.task_done()
            except queue.Empty: break
            except Exception as e: logging.warning(f"Error clearing item from queue during reset: {e}")
        logging.debug("State reset for new check run.")

        # --- 1. Preprocessing, Filtering, and KG Key Generation ---
        logging.info("Step 1: Preprocessing, Filtering, and KG Key Generation...");
        try:
            # Returns list of dicts for checkable claims, list of strings for non-checkable
            checkable_claims_initial_data, non_checkable_sents = self.preprocess_and_filter(text)
        except Exception as e:
            logging.critical(f"Fatal Error during Preprocessing/Filtering: {e}", exc_info=True)
            return {"processed_claims":[], "non_checkable_claims":[f"Error during preprocessing: {e}"], "summary":f"Fatal Error during Preprocessing: {e}", "raw_fact_checks":{}, "raw_searches":{}, "shap_explanations": []}

        # If no checkable claims found, return early
        if not checkable_claims_initial_data:
            logging.warning("No checkable claims found after preprocessing and filtering.");
            summary = self.generate_chain_of_thought([], non_checkable_sents) # Generate summary based on filtering
            return {"processed_claims":[], "non_checkable_claims":non_checkable_sents, "summary":summary, "raw_fact_checks":{}, "raw_searches":{}, "shap_explanations": []}

        # --- 2. Prioritize Checkable Claims ---
        logging.info("Step 2: Prioritizing Checkable Claims...");
        try:
            prioritized_claims_data = self.classify_and_prioritize_claims(checkable_claims_initial_data)
        except Exception as e:
             logging.error(f"Prioritization failed: {e}. Proceeding without priority sorting.", exc_info=True);
             prioritized_claims_data = checkable_claims_initial_data # Fallback to original order

        if not prioritized_claims_data: # Should not happen if initial data existed, but check defensively
            logging.error("Prioritization resulted in empty list unexpectedly.");
            summary = self.generate_chain_of_thought([], non_checkable_sents)
            return {"processed_claims":[], "non_checkable_claims":non_checkable_sents, "summary":"Prioritization failed, no claims available.", "raw_fact_checks":{}, "raw_searches":{}, "shap_explanations": []}

        # --- 3. Check Knowledge Graph (Neo4j) ---
        logging.info("Step 3: Checking Knowledge Graph (Neo4j)...");
        claims_to_process_fully = []
        claims_found_in_kg = []
        claims_with_errors_pre_pipeline = [] # Store claims that failed basic checks before queueing

        if not self.neo4j_driver:
            logging.warning("Neo4j driver unavailable. Skipping KG check for all claims.")
            claims_to_process_fully = prioritized_claims_data # All claims need full processing if KG fails
        else:
            for claim_data_dict in prioritized_claims_data:
                 if not isinstance(claim_data_dict, dict):
                      logging.error(f"Invalid item encountered in prioritized claims data: {claim_data_dict}. Skipping.")
                      claims_with_errors_pre_pipeline.append({"original_claim": "Invalid Data Item", "source": "Error", "final_label": "Processing Error", "confidence": 0.0, "final_explanation": "Invalid data structure."})
                      continue

                 preprocessed_claim_text = claim_data_dict.get('preprocessed_claim')
                 original_claim = claim_data_dict.get('original_claim', '?')

                 # Validate essential data needed for KG check
                 if not preprocessed_claim_text or original_claim == '?':
                      logging.warning(f"Skipping KG check for claim due to missing preprocessed or original text: '{original_claim[:50]}...'")
                      claim_data_dict['source'] = 'Error'
                      claim_data_dict['final_label'] = 'Preprocessing Error'
                      claim_data_dict['confidence'] = 0.1
                      claim_data_dict['final_explanation'] = 'Missing essential text fields before KG check.'
                      claims_with_errors_pre_pipeline.append(claim_data_dict)
                      continue # Don't check KG or queue

                 # Perform the KG lookup
                 kg_result = self.check_kg_for_claim(preprocessed_claim_text)
                 if kg_result:
                     claims_found_in_kg.append(kg_result) # Store the result from KG
                 else:
                     claims_to_process_fully.append(claim_data_dict) # Add to list for full pipeline

        logging.info(f"KG Check Complete: Found={len(claims_found_in_kg)}, Needs Full Processing={len(claims_to_process_fully)}, Pre-Pipeline Errors={len(claims_with_errors_pre_pipeline)}")

        # --- 4. Process Claims via Worker Threads (if any need full processing) ---
        if claims_to_process_fully:
            logging.info(f"Step 4a: Queueing {len(claims_to_process_fully)} claims for full processing...");
            self.add_claims_to_queue(claims_to_process_fully)

            logging.info("Step 4b: Starting worker threads for parallel processing...");
            threads = []; n_cpu = os.cpu_count() or 1;
            # Determine number of workers: min(requested, queue_size, cpu_cores)
            n_workers = min(num_workers, self.claim_queue.qsize(), n_cpu if n_cpu else 1)

            if n_workers > 0:
                logging.info(f"Launching {n_workers} worker thread(s)...");
                for i in range(n_workers):
                    t = Thread(target=self.worker, name=f"Worker-{i+1}", daemon=True);
                    t.start();
                    threads.append(t)

                # Wait for all tasks in the queue to be processed
                self.claim_queue.join();
                logging.info("All claims processed by workers (queue is empty).")

                # Optionally signal workers to stop (using None sentinel) - not strictly needed with daemon threads if join works
                # for _ in range(n_workers): self.claim_queue.put(None)

                # Wait for worker threads to finish
                for t in threads:
                    t.join(timeout=10.0) # Add a timeout to join
                    if t.is_alive(): logging.warning(f"Thread {t.name} did not terminate gracefully after join timeout!")
                logging.info("Worker threads finished.")
            else:
                 logging.info("Worker processing skipped (queue was empty or num_workers set to 0).")
        else:
            logging.info("Step 4: Skipping full pipeline processing (no claims needed it).")

        # --- 5. Generate SHAP Explanations (if available and needed) ---
        logging.info("Step 5: Generating SHAP explanations (if available)...");
        # Run SHAP only on the claims that actually went through the full pipeline
        self.train_and_run_shap(claims_to_process_fully)

        # --- 6. Consolidate Final Results ---
        logging.info("Step 6: Consolidating final results...");
        final_results_list = []
        processed_count = 0
        error_count = 0

        # Add results from KG hits
        final_results_list.extend(claims_found_in_kg)
        processed_count += len(claims_found_in_kg)

        # Add results from pre-pipeline errors
        final_results_list.extend(claims_with_errors_pre_pipeline)
        processed_count += len(claims_with_errors_pre_pipeline)
        error_count += len(claims_with_errors_pre_pipeline)

        # Retrieve results from the shared dictionary for claims processed by workers
        with self.results_lock:
            for claim_data_dict in claims_to_process_fully:
                 original_claim = claim_data_dict.get('original_claim')
                 if not original_claim: continue # Should not happen, but safeguard

                 pipeline_result = self.results.get(original_claim)
                 if pipeline_result:
                      # Avoid duplicates if somehow added earlier (e.g., error fallback)
                      if not any(res.get('original_claim') == original_claim for res in final_results_list):
                           final_results_list.append(pipeline_result)
                           processed_count += 1
                           # Count errors from the pipeline stage
                           if pipeline_result.get("source") == "Error" or "Error" in pipeline_result.get("final_label",""):
                               error_count += 1
                 else:
                      # This indicates a problem - worker finished but result missing
                      logging.error(f"Result missing from self.results for fully processed claim: '{original_claim}'. This indicates a potential bug or lost result.")
                      # Add an error entry to signify the loss
                      error_entry = {
                          "original_claim": original_claim,
                          "preprocessed_claim": claim_data_dict.get('preprocessed_claim','?'),
                          "ner_entities": claim_data_dict.get('ner_entities', []),
                          "factual_score": claim_data_dict.get('factual_score', 0.0),
                          "final_label": "Missing Result",
                          "confidence": 0.0,
                          "final_explanation": "Result was lost after worker processing. Check logs for worker errors.",
                          "source": "Error"
                      }
                      if not any(res.get('original_claim') == original_claim for res in final_results_list):
                           final_results_list.append(error_entry)
                           processed_count += 1; error_count += 1

        # --- 7. Generate Final Summary (Chain of Thought) ---
        logging.info("Step 7: Generating final summary...");
        summary = self.generate_chain_of_thought(final_results_list, non_checkable_sents)

        # --- 8. Final Logging and Return ---
        duration = time.time() - start_time
        pipeline_ok_count = len([r for r in final_results_list if r.get('source') == 'Full Pipeline' and r.get('final_label') not in ["Error", "LLM Error", "Processing Error"]])
        logging.info(
            f"Check Complete. Total Initial Checkable={len(checkable_claims_initial_data)}, "
            f"KG Hits={len(claims_found_in_kg)}, Pipeline OK={pipeline_ok_count}, "
            f"Total Errors/Warnings={error_count} in {duration:.2f}s."
        )

        return {
            "processed_claims": final_results_list,
            "non_checkable_claims": non_checkable_sents,
            "summary": summary,
            "raw_fact_checks": self.raw_fact_checks,
            "raw_searches": self.raw_searches,
            "shap_explanations": self.shap_explanations
            }

    def close_neo4j(self):
        """Closes the Neo4j driver connection if it's open."""
        if self.neo4j_driver:
            try:
                self.neo4j_driver.close()
                logging.info("Neo4j driver closed successfully.")
            except Exception as e:
                logging.error(f"Error closing Neo4j driver: {e}", exc_info=True)
            finally:
                 self.neo4j_driver = None # Ensure driver is marked as closed

# --- Main Execution Block ---
if __name__ == "__main__":
    print("--- VeriStream Fact Checker ---")
    # Dependency Checks
    if shap is None: print("\n[WARN] SHAP library not installed (pip install shap). Explainability disabled.")
    else: print("[INFO] SHAP library found.")
    if spacy is None: print("\n[WARN] spaCy library not installed (pip install spacy). NLP features limited.")
    elif NLP is None: print("\n[WARN] spaCy model 'en_core_web_sm' not found (python -m spacy download en_core_web_sm). NLP features limited.")
    else: print("[INFO] spaCy library and model found.")
    if Groq is None: print(f"\n[WARN] Groq library not installed (pip install groq). LLM ({LLM_PROVIDER}) disabled.")
    elif groq_client is None: print(f"\n[WARN] Groq client initialization failed (Check GROQ_API_KEY env var). LLM ({LLM_PROVIDER}) disabled.")
    else: print(f"[INFO] {LLM_PROVIDER} client available (Model: {LLM_MODEL_NAME}).")

    # Configuration Checks
    print("\n[INFO] Checking Configuration...")
    if not GOOGLE_API_KEY or not GOOGLE_CSE_ID:
        logging.critical("CRITICAL: Google API Key or CSE ID missing in environment variables.")
        print("\n[CRITICAL ERROR] Google API Key or CSE ID missing. Please set GOOGLE_API_KEY and GOOGLE_CSE_ID. Exiting.")
        exit(1)
    else: logging.info("Google API Key and CSE ID loaded.")

    if not NEO4J_URI or not NEO4J_USER or not NEO4J_PASSWORD:
         logging.warning("Neo4j connection details (URI, USER, PASSWORD) missing. Knowledge Graph features will be disabled.")
         print("\n[WARN] Neo4j connection details missing. KG features disabled.")
    else: logging.info("Neo4j credentials loaded.")

    # Initialize FactChecker
    print("[INFO] Initializing FactChecker...")
    checker = None
    try:
        checker = FactChecker()
        # Check if Neo4j initialization succeeded within the checker
        if checker.neo4j_driver is None and (NEO4J_URI and NEO4J_USER and NEO4J_PASSWORD):
             print("[WARN] FactChecker initialized, but Neo4j connection failed. KG features disabled.")
        elif checker.neo4j_driver:
             print("[INFO] FactChecker initialized successfully with Neo4j connection.")
        else:
             print("[INFO] FactChecker initialized successfully (Neo4j disabled due to missing config).")

    except RuntimeError as e: # Catch model loading errors specifically
        logging.critical(f"FactChecker initialization failed (Likely model loading error): {e}", exc_info=True)
        print(f"\n[CRITICAL ERROR] Initialization failed. Check model '{EMBEDDING_MODEL_NAME}' availability. Log: {log_file}. Error: {e}")
        exit(1)
    except Exception as e: # Catch other unexpected errors
        logging.critical(f"Unexpected error during FactChecker initialization: {e}", exc_info=True)
        print(f"\n[CRITICAL UNEXPECTED ERROR] during initialization. Check logs: {log_file}. Error: {e}")
        exit(1)

    # --- Input Text ---
    input_text = (
        "The Eiffel Tower is located in Berlin. Fact checkers say this is false. "
        "I think Paris is the most beautiful city. COVID-19 vaccines are ineffective according to some studies. "
        "Water boils at 100 degrees Celsius at sea level. This statement is true. "
        "The earth is flat according to some sources. Llamas are native to North America, not South America. "
        "Quantum computing will break all encryption soon. This sentence should be ignored."
        "Is VeriStream the best fact-checker?"
    )
    print(f"\n--- Input Text to Check ---\n{input_text}\n" + "-"*27)

    # --- Run Check ---
    print("\n--- Starting Fact Check Pipeline ---")
    if not checker: print("[ERROR] Checker not initialized. Exiting."); exit(1) # Should not happen, but safeguard

    results_data = None
    try:
        # Run the main check method
        results_data = checker.check(input_text, num_workers=2)
    except Exception as check_e:
        logging.critical(f"Critical error during checker.check() execution: {check_e}", exc_info=True)
        print(f"\n[CRITICAL ERROR] during fact checking process. Check logs: {log_file}. Error: {check_e}")
        if checker: checker.close_neo4j() # Attempt cleanup
        exit(1)

    # --- Output Results ---
    print("\n--- Fact Check Results ---")

    # 1. Raw Google Fact Check API Results (Optional Detail)
    print("\n" + "="*25 + " Intermediate Output 1: Raw Google Fact Check API Results " + "="*15)
    raw_checks = results_data.get("raw_fact_checks", {})
    if raw_checks:
        print(" (Note: Shows results only for claims requiring full API processing, using original claim as query)")
        for claim in sorted(raw_checks.keys()): # Sort claims for consistent output
            api_res_list = raw_checks[claim]
            print(f"\nClaim (Original Query): \"{claim}\"")
            if api_res_list and isinstance(api_res_list, list):
                for i, res_item in enumerate(api_res_list):
                    print(f"  Result {i+1}: Verdict='{res_item.get('verdict','?')}', Evidence='{res_item.get('evidence','?')}'")
            elif isinstance(api_res_list, dict) and api_res_list.get('verdict') == 'Error': # Handle single error dict
                 print(f"  Error: Verdict='{api_res_list.get('verdict','?')}', Evidence='{api_res_list.get('evidence','?')}'")
            else: print(f"  - No result stored or invalid format: {api_res_list}")
    else: print("  - No Fact Check API calls were made or results stored (e.g., all KG hits or errors before API call).")
    print("="*81)

    # 2. Raw Google Custom Search Results (Optional Detail)
    SHOW_RAW_SEARCH = False # Set to True to see raw search snippets
    raw_searches = results_data.get("raw_searches", {})
    if SHOW_RAW_SEARCH and raw_searches:
        print("\n" + "="*25 + " Intermediate Output 2: Raw Google Custom Search Snippets " + "="*14)
        print(" (Note: Shows results only for claims requiring RAG, using original claim as query)")
        for claim in sorted(raw_searches.keys()): # Sort claims
            search_data = raw_searches[claim]
            print(f"\nClaim (Original Query): \"{claim}\"")
            search_results = search_data.get("results", [])
            if search_results:
                for i, item in enumerate(search_results):
                    print(f"  {i+1}. Title: {item.get('title','?')}")
                    print(f"     Snippet: {item.get('snippet','?')}")
                    print(f"     Link: {item.get('link','?')}")
            elif search_data.get("response") is not None: print("  - Search API call succeeded, but no items were returned.")
            else: print("  - Search API call likely failed or response structure was missing.")
        print("="*81)

    # 3. Filtered Non-Checkable Sentences
    print("\n" + "="*25 + " Preprocessing Output: Filtered Non-Checkable Sentences " + "="*16)
    non_checkable = results_data.get("non_checkable_claims", [])
    if non_checkable:
        for i, claim in enumerate(sorted(non_checkable)): # Sort for consistency
            print(f"  {i+1}. \"{claim}\"")
    else: print("  - No sentences were filtered out as non-checkable.")
    print("="*81)

    # 4. Final Processed Claim Details (Main Output)
    print("\n" + "="*30 + " Final Processed Claim Details " + "="*30)
    processed_claims = results_data.get("processed_claims", [])
    if processed_claims:
        # Sort final results by the original claim text for consistent display
        sorted_results = sorted(processed_claims, key=lambda x: x.get('original_claim', ''))
        for i, res in enumerate(sorted_results):
            # Extract all relevant fields with defaults
            source = res.get('source', 'Unknown')
            final_label = res.get('final_label', 'N/A')
            confidence = res.get('confidence', 0.0)
            explanation = res.get('final_explanation', 'N/A')
            original_claim = res.get('original_claim', '?')
            preprocessed_claim = res.get('preprocessed_claim', 'N/A') # Key used for KG
            factual_score = res.get('factual_score') # May be None if not calculated
            initial_verdict = res.get('initial_verdict_raw', 'N/A')
            initial_evidence = res.get('initial_evidence', 'N/A')
            rag_status = res.get('rag_status', 'N/A')
            ner_entities = res.get('ner_entities', [])
            top_snippets = res.get('top_rag_snippets', []) # Formatted strings
            kg_timestamp = res.get('kg_timestamp') # Timestamp from KG if applicable

            print(f"\n--- Claim {i+1} ---")
            print(f"Original Claim: \"{original_claim}\"")
            print(f"Processing Source: {source}")

            if source == "Full Pipeline":
                print(f"  - KG Match Key (Preprocessed): \"{preprocessed_claim}\"")
                if ner_entities: print(f"  - NER Entities: {', '.join(['{}({})'.format(e['text'], e['label']) for e in ner_entities])}")
                else: print("  - NER Entities: None Found or Extracted")
                print(f"  - Priority Score (Heuristic): {factual_score:.3f}" if factual_score is not None else "N/A")
                print(f"  - Initial API Check: Verdict='{initial_verdict}', Source='{initial_evidence}'")
                print(f"  - RAG Status: {rag_status}")
                if top_snippets:
                    print("  - Top RAG Snippets Retrieved:")
                    for j, snip in enumerate(top_snippets): print(f"    {j+1}. {snip}")
                else: print("  - Top RAG Snippets Retrieved: None")
                print(f"  - Final Verdict (LLM Synthesized): {final_label}")
                print(f"  - Confidence: {confidence:.2f}")
                print(f"  - LLM Justification: {explanation}")
                if checker and checker.neo4j_driver: print(f"  - Neo4j Storage Status: Stored/Updated in DB '{NEO4J_DATABASE}'")

            elif source == "Knowledge Graph":
                print(f"  - KG Match Key (Preprocessed): \"{preprocessed_claim}\"")
                print(f"  - Final Verdict (From KG): {final_label}")
                print(f"  - Confidence (From KG): {confidence:.2f}")
                print(f"  - KG Explanation: {explanation}")
                if kg_timestamp: print(f"  - KG Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime(kg_timestamp))}")

            elif source == "Error":
                print(f"  - Final Verdict: {final_label}")
                print(f"  - Confidence: {confidence:.2f}")
                print(f"  - Error Explanation: {explanation}")
            else: # Should not happen
                 print(f"  - Status: Unknown processing source '{source}' or incomplete result format.")
                 print(f"  - Details: Label={final_label}, Conf={confidence:.2f}, Explain={explanation}")
            print("-" * 15) # Separator for claims

    else: print("  - No checkable claims were processed or results available.")
    print("="*83)

    # 5. XAI (SHAP) Summary
    print("\n" + "="*30 + " XAI (SHAP) Summary " + "="*37)
    shap_explanations = results_data.get("shap_explanations", [])
    if shap_explanations:
         # Filter to show only SHAP results for claims that *should* have them (processed fully)
         fully_processed_claims_texts = {p['original_claim'] for p in processed_claims if p.get('source') == 'Full Pipeline'}
         relevant_explanations = [ex for ex in shap_explanations if ex.get('claim') in fully_processed_claims_texts]

         if relevant_explanations:
             print(f" (Note: Shows SHAP status for {len(relevant_explanations)} claim(s) processed via full pipeline)")
             shap_summary_lines=[]; has_real_values=False; has_errors=False; all_zero=True

             # Sort by claim for consistent output
             relevant_explanations_sorted = sorted(relevant_explanations, key=lambda x: x.get('claim', ''))
             for expl in relevant_explanations_sorted:
                 claim_text = expl.get('claim', '?')[:40] + '...'
                 v = expl.get('shap_values', [])
                 s = "[Status Unknown]" # Default status string

                 # Determine status string based on SHAP values content
                 if isinstance(v, list) and v:
                     if any(isinstance(val, str) and val.startswith('[SHAP') for val in v):
                         s = v[0] if v and isinstance(v[0], str) else "[SHAP Err Fallback]"
                         has_errors = True; all_zero = False
                     elif all(isinstance(x, (int, float)) and abs(float(x)) < 1e-9 for x in v):
                         s = "[Zero Values]"
                     elif all(isinstance(x, (int, float)) for x in v):
                         s = f"[~{len(v)} numerical values]"
                         has_real_values = True; all_zero = False
                     else:
                         s = "[Mixed/Invalid Data]"; all_zero = False
                 elif isinstance(v, str) and v.startswith('[SHAP'):
                     s = v; has_errors = True; all_zero = False
                 elif not v:
                     s = "[No SHAP Data]"; all_zero = False

                 shap_summary_lines.append(f"  - '{claim_text}': {s}")

             # Determine overall SHAP status message
             status = "Unavailable/Not Run."
             if has_real_values: status = "Generated numerical values."
             elif has_errors: status = "Failed/Unavailable (Errors encountered)."
             elif all_zero and not has_errors and relevant_explanations: status = "Zero values reported (Prediction function likely unsuitable or insensitive)."
             elif relevant_explanations: status = "Completed (No non-zero values or errors detected)."

             print(f"\n  Overall SHAP Status: {status}")
             if shap_summary_lines:
                  print("  Claim-Specific Status:")
                  print("\n".join(shap_summary_lines))

             # Add warning messages based on status
             if has_errors: print(f"\n  *** SHAP Error Detected: Check '{log_file}' for detailed SHAP errors. ***")
             elif all_zero and relevant_explanations: print(f"\n  *** SHAP produced zero values: The current prediction function (embedding norm) may be too simple for SHAP analysis. Check '{log_file}'. ***")

         elif checker and checker.shap_available:
             print("  - SHAP analysis not applicable: No claims required full processing, or SHAP failed before generating results.")
         else: # SHAP library not installed
              print("  - SHAP analysis skipped (library not installed).")
    else:
         print("  - SHAP analysis results structure missing or empty.")
    print("="*86)

    # 6. Chain of Thought Summary
    print("\n" + "="*30 + " Chain of Thought Summary " + "="*30)
    print(results_data.get("summary", "No summary generated."))
    print("="*86)

    # --- Cleanup ---
    print(f"\nLog file generated at: {log_file}")
    if checker:
        checker.close_neo4j() # Close Neo4j connection cleanly

    print("\nScript finished.")
