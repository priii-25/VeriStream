#backend/fact_checker.py
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
NUM_SEARCH_RESULTS = 3 # Keep lower
RAG_K = 3
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

# --- API Functions ---
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

# --- NEW LLM Final Verdict Function (Based on RAG) ---
def get_llm_final_verdict_from_rag(claim: str, rag_evidence: list, rag_status_msg: str) -> dict:
    """Uses Groq LLM to determine a final True/False verdict based *only* on RAG evidence."""
    if not groq_client: return {"final_label": "LLM Error", "explanation": "Groq client not initialized.", "confidence": 0.1}

    rag_snippets_formatted = f"\nRetrieved Web Snippets (RAG Status: {rag_status_msg}):\n"
    if rag_evidence:
        for i, doc in enumerate(rag_evidence): rag_snippets_formatted += f"{i+1}. Snippet: \"{doc.get('content','N/A')}\"\n   Source: {doc.get('metadata',{}).get('source','N/A')}\n"
    else: rag_snippets_formatted += "None Provided or Retrieval Failed.\n"

    # Prompt focuses ONLY on claim and RAG snippets
    prompt = f"""You are a fact-checker. Evaluate the 'Claim' below based *only* on the provided 'Retrieved Web Snippets (RAG Evidence)'. Ignore any initial checks mentioned elsewhere.

Claim: "{claim}"
{rag_snippets_formatted}
Evaluation Task: Based *strictly* on the RAG evidence provided above, determine if the claim is True or False.
- If the RAG snippets strongly and consistently support the claim, verdict is "True".
- If the RAG snippets contradict the claim, verdict is "False".
- If RAG evidence retrieval failed, returned no relevant snippets, or the snippets are clearly insufficient/irrelevant to judge the claim's factuality, the verdict *must* be "False" (indicating lack of verification from web search).

Output Format: Respond with *only* a valid JSON object containing exactly two keys:
1. "verdict": Your verdict. Must be the string "True" or the string "False".
2. "justification": A brief explanation (1-2 sentences) citing *only* the RAG evidence (or lack thereof) that led to your verdict.

Example (Insufficient RAG):
{{
  "verdict": "False",
  "justification": "RAG evidence retrieval failed (or snippets were irrelevant), so the claim could not be verified from web search."
}}

Your JSON Response:
"""
    logging.debug(f"LLM Final Verdict Prompt for '{claim}':\n{prompt}")
    try:
        response = groq_client.chat.completions.create(model=LLM_MODEL_NAME, messages=[{"role":"system","content":"Output only valid JSON."}, {"role":"user","content":prompt}], temperature=0.0, max_tokens=150)
        llm_output_text = response.choices[0].message.content; logging.debug(f"Groq LLM Raw: {llm_output_text}")
        # Parse JSON
        try:
            # Handle potential markdown code blocks
            if llm_output_text.strip().startswith("```json"): llm_output_text=llm_output_text.strip()[7:-3].strip()
            elif llm_output_text.strip().startswith("```"): llm_output_text=llm_output_text.strip()[3:-3].strip()

            llm_res = json.loads(llm_output_text); v=llm_res.get("verdict"); j=llm_res.get("justification")
            if isinstance(v,str) and v in ["True","False"] and isinstance(j,str) and j:
                conf = 0.90 if v=="True" else (0.88 if rag_evidence else 0.85) # Slightly lower confidence if False based on no evidence
                return {"final_label":v, "explanation":j, "confidence":conf}
            else: logging.error(f"LLM invalid content: {llm_output_text}"); return {"final_label":"LLM Error", "explanation":f"Invalid LLM JSON content: {llm_output_text}", "confidence":0.1}
        except json.JSONDecodeError as e: logging.error(f"LLM JSON decode fail: {e}. Resp: {llm_output_text}"); return {"final_label":"LLM Error", "explanation":f"LLM non-JSON: {llm_output_text}", "confidence":0.1}
        except Exception as e: logging.error(f"LLM parse err: {e}. Resp: {llm_output_text}"); return {"final_label":"LLM Error", "explanation":f"LLM parse err: {llm_output_text}", "confidence":0.1}
    except APIConnectionError as e: logging.error(f"Groq ConnErr: {e}"); return {"final_label":"LLM Error", "explanation":f"API Conn Err: {e}", "confidence":0.1}
    except RateLimitError as e: logging.error(f"Groq RateLimit: {e}"); time.sleep(5); return {"final_label":"LLM Error", "explanation":f"API Rate Limit: {e}", "confidence":0.1}
    except APIStatusError as e: logging.error(f"Groq Status {e.status_code}: {e.response}"); return {"final_label":"LLM Error", "explanation":f"API Status {e.status_code}", "confidence": 0.1} # Corrected log message
    except Exception as e: logging.error(f"Unexpected Groq API err: {e}", exc_info=True); return {"final_label":"LLM Error", "explanation":f"API Err: {e}", "confidence":0.1}

# --- Helper Functions ---
def is_claim_checkable(sentence: str) -> bool:
    # (Updated logic from previous answer)
    if not sentence or not isinstance(sentence, str): return False
    text_lower = sentence.lower().strip()
    if text_lower.startswith('"') and text_lower.endswith('"'): text_lower = text_lower[1:-1]
    if text_lower.startswith("'") and text_lower.endswith("'"): text_lower = text_lower[1:-1]
    for phrase in OPINION_PHRASES:
        if text_lower.startswith(phrase+" "): logging.info(f"Filtered (Opinion): '{sentence}'"); return False
    if any(word in text_lower for word in SELF_REFERENCE_WORDS): logging.info(f"Filtered (Self-Ref): '{sentence}'"); return False
    if text_lower.endswith("?"): logging.info(f"Filtered (Question): '{sentence}'"); return False
    if NLP:
        doc = NLP(sentence)
        if any(t.lemma_.lower() in SUBJECTIVE_ADJECTIVES for t in doc if t.pos_=="ADJ"): logging.info(f"Filtered (Subjective): '{sentence}'"); return False
        has_verb = any(t.pos_ in ("VERB","AUX") for t in doc); has_subj = any(t.dep_ in ("nsubj","nsubjpass","csubj","csubjpass","expl") for t in doc)
        if not (has_verb and has_subj) and len(doc)<4: logging.info(f"Filtered (Structure): '{sentence}'"); return False
        if len(doc)>0 and doc[0].pos_=="VERB" and doc[0].tag_=="VB":
             is_imperative = not any(t.dep_.startswith("nsubj") for t in doc)
             if is_imperative: logging.info(f"Filtered (Imperative): '{sentence}'"); return False
    return True

def preprocess_claim_for_api(original_claim: str) -> str:
    # (Same preprocessing logic as before)
    if not NLP or not original_claim: return original_claim
    try:
        doc = NLP(original_claim); simplified_parts = []; root = None; subj = None; obj_or_comp = None; negation = False
        for token in doc:
            if token.dep_ == "ROOT": root = token
            if token.dep_ == "neg": negation = True
        if not root:
             if len(doc) < 5: return original_claim
             else: logging.warning(f"No ROOT for '{original_claim}', using original."); return original_claim
        for child in root.children:
            if "subj" in child.dep_: subj = child
            elif "obj" in child.dep_ or "attr" in child.dep_ or "acomp" in child.dep_:
                 obj_or_comp = child
                 while True: # Try to find a more substantive head in the phrase
                      potential_heads = [c for c in obj_or_comp.children if c.pos_ in ("NOUN", "PROPN", "ADJ", "NUM")]
                      if potential_heads: obj_or_comp = potential_heads[0]; continue # Go deeper if possible
                      else: break # Use current obj_or_comp
        if subj: simplified_parts.append(subj.text)
        if negation: simplified_parts.append("not")
        if root.pos_ != 'NOUN': # Avoid using nouns like 'capital' as the main verb if possible
            simplified_parts.append(root.lemma_)
        elif root.head.pos_ == 'VERB': # If root is noun but head is verb (e.g. auxiliary), use head verb
             simplified_parts.append(root.head.lemma_)
        # Add aux verb if needed (like 'is')
        aux_verb = next((t for t in root.children if t.dep_ == 'aux' or (t.dep_=='attr' and t.pos_=='AUX')), None)
        if not root.lemma_ in simplified_parts and aux_verb: simplified_parts.append(aux_verb.lemma_)

        if obj_or_comp: simplified_parts.append(obj_or_comp.text)
        simplified_claim = " ".join(simplified_parts)
        if len(simplified_claim.split()) < 2 or not simplified_claim: logging.warning(f"Short simplification for '{original_claim}', using original."); return original_claim
        logging.info(f"Simplified '{original_claim}' -> '{simplified_claim}' for API."); return simplified_claim
    except Exception as e: logging.error(f"Error preprocessing '{original_claim}': {e}"); return original_claim

# --- FactChecker Class ---
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
        self.raw_fact_checks={}; self.raw_searches={} # Removed self.claim_ner (handled within claim data dict)

        # Neo4j Driver Initialization
        self.neo4j_driver = None
        try:
            # Use basic_auth and specify database
            self.neo4j_driver = GraphDatabase.driver(NEO4J_URI, auth=basic_auth(NEO4J_USER, NEO4J_PASSWORD))
            # Verify connection and database existence
            with self.neo4j_driver.session(database=NEO4J_DATABASE) as session:
                session.run("RETURN 1") # Simple query to test connection
            logging.info(f"Neo4j driver initialized for database '{NEO4J_DATABASE}'.")
        except neo4j_exceptions.AuthError as auth_err:
             logging.error(f"Neo4j Authentication Failed for user '{NEO4J_USER}'. Check credentials. Error: {auth_err}")
             self.neo4j_driver = None
        except neo4j_exceptions.ServiceUnavailable as conn_err:
             logging.error(f"Neo4j Service Unavailable at URI '{NEO4J_URI}'. Check if Neo4j is running. Error: {conn_err}")
             self.neo4j_driver = None
        except Exception as e:
            logging.error(f"Failed to initialize Neo4j driver for URI '{NEO4J_URI}', DB '{NEO4J_DATABASE}': {e}", exc_info=True)
            self.neo4j_driver = None # Ensure driver is None if init fails


    # --- ADDED: Function to check Knowledge Graph first ---
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
               c.timestamp AS timestamp // Get timestamp for potential freshness check later
        ORDER BY v.confidence DESC, c.timestamp DESC // Prioritize higher confidence, then newer
        LIMIT 1
        """
        params = {"prep_text": preprocessed_claim, "min_confidence": KG_CONFIDENCE_THRESHOLD}

        try:
            with self.neo4j_driver.session(database=NEO4J_DATABASE) as session:
                result = session.run(query, params).single()

            if result:
                logging.info(f"KG HIT: Found existing verdict for '{preprocessed_claim}' -> '{result['verdict_label']}' (Conf: {result['confidence']:.2f})")
                # Format the result similarly to the final pipeline output
                kg_result_dict = {
                    "original_claim": result["original_claim"],
                    "preprocessed_claim": preprocessed_claim, # Include for consistency
                    "ner_entities": [], # Not fetched in this simple query, could be added
                    "factual_score": None, # Not fetched
                    "initial_verdict_raw": "From KG",
                    "initial_evidence": "From KG",
                    "rag_status": "N/A (From KG)",
                    "top_rag_snippets": [], # Not applicable
                    "final_verdict": f"{result['verdict_label']} (Confidence: {result['confidence']:.2f})",
                    "final_explanation": result["explanation"],
                    "source": "Knowledge Graph", # Add source field
                    "kg_timestamp": result["timestamp"] # Store timestamp if needed
                }
                return kg_result_dict
            else:
                logging.info(f"KG MISS: No reliable verdict found for '{preprocessed_claim}'.")
                return None
        except neo4j_exceptions.ServiceUnavailable as e:
            logging.error(f"KG Check Failed: Neo4j connection error: {e}")
            return None # Treat connection errors as a miss
        except Exception as e:
            logging.error(f"KG Check Failed: Error querying Neo4j for '{preprocessed_claim}': {e}", exc_info=True)
            return None # Treat other errors as a miss


    # --- UPDATED store_in_neo4j function ---
    def store_in_neo4j(self, claim_data):
        if not self.neo4j_driver:
           logging.error("Neo4j driver not initialized. Cannot store data.")
           return
        # Avoid storing if the source was already the KG
        if claim_data.get("source") == "Knowledge Graph":
             logging.debug(f"Skipping Neo4j store for claim already retrieved from KG: '{claim_data['original_claim']}'")
             return

        claim = claim_data['original_claim']
        preprocessed_claim = claim_data['preprocessed_claim']
        final_verdict_str = claim_data['final_verdict'] # e.g., "True (Confidence: 0.90)"
        final_explanation = claim_data['final_explanation']
        confidence = 0.0
        final_label = "Error" # Default label
        try:
            if '(Confidence: ' in final_verdict_str:
                parts = final_verdict_str.split('(Confidence: ')
                final_label = parts[0].strip() # e.g., "True" or "False"
                confidence_str = parts[1][:-1] # Remove closing parenthesis
                confidence = float(confidence_str)
            elif final_verdict_str in ["True", "False"]: # Handle simple True/False if confidence parsing failed but label is valid
                 final_label = final_verdict_str
                 confidence = 0.5 # Assign default confidence? Or keep 0?
                 logging.warning(f"Could not parse confidence from final verdict string: '{final_verdict_str}'. Using label '{final_label}' and default confidence.")
            else: # Handle error cases like "LLM Error", "Processing Error"
                 final_label = final_verdict_str # Store the error string as the label
                 confidence = 0.1 # Low confidence for errors
                 logging.warning(f"Storing non-standard verdict label: '{final_verdict_str}'")
        except Exception as e:
            final_label = final_verdict_str # In case parsing fails, keep the original string
            confidence = 0.1
            logging.warning(f"Could not parse confidence from final verdict string '{final_verdict_str}': {e}")


        entities = claim_data.get('ner_entities', []) # Use .get for safety
        initial_evidence = claim_data.get('initial_evidence', "")
        rag_status = claim_data.get('rag_status', "")
        initial_verdict_raw = claim_data.get('initial_verdict_raw', "")
        # Use get with default for factual_score
        factual_score = claim_data.get('factual_score', 0.0)
        if factual_score is None: factual_score = 0.0 # Ensure it's not None

        top_rag_snippets = claim_data.get('top_rag_snippets', []) # List of formatted snippet strings
        timestamp = time.time()

        # Use a session specific to the target database
        with self.neo4j_driver.session(database=NEO4J_DATABASE) as session:
            tx = None # Initialize tx outside try
            try:
                tx = session.begin_transaction()

                # 1. MERGE Claim node based on preprocessed_text to avoid duplicates
                # If claim exists, update timestamp and other potentially changed props.
                # If not exists, create it.
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
                        c.timestamp = $timestamp, // Update timestamp on match
                        c.initial_verdict_raw = $initial_verdict_raw,
                        c.rag_status = $rag_status,
                        c.initial_evidence = $initial_evidence,
                        c.factual_score = $factual_score,
                        // Optionally update original text if it differs significantly?
                        c.text = CASE WHEN c.text <> $text THEN $text ELSE c.text END
                    RETURN c, id(c) as claim_id
                """,
                                 text=claim, preprocessed_text=preprocessed_claim, timestamp=timestamp,
                                 initial_verdict_raw=initial_verdict_raw, rag_status=rag_status,
                                 initial_evidence=initial_evidence, factual_score=factual_score).single()

                if not claim_node_result:
                    raise Exception(f"Claim node MERGE failed for preprocessed text: {preprocessed_claim}")

                claim_node = claim_node_result['c']      # The actual node object
                claim_node_id = claim_node_result['claim_id'] # The internal Neo4j ID

                # --- Verdict Handling: Remove old verdicts, create new one ---
                # It's often cleaner to remove old related nodes like verdicts/evidence
                # and create new ones rather than trying to update them in place,
                # especially if the verdict could change significantly.

                # Delete existing HAS_VERDICT relationship and Verdict node for this claim
                tx.run("""
                    MATCH (c:Claim)-[r:HAS_VERDICT]->(old_v:Verdict)
                    WHERE id(c) = $claim_node_id
                    DELETE r, old_v
                """, claim_node_id=claim_node_id)

                # 2. Create the NEW Verdict node
                verdict_node_result = tx.run("""
                    CREATE (v:Verdict {
                        verdict_label: $verdict_label,
                        confidence: $confidence,
                        explanation: $explanation
                    })
                    RETURN id(v) as verdict_id
                    """, verdict_label=final_label, confidence=confidence, explanation=final_explanation).single()

                if not verdict_node_result:
                    raise Exception("Verdict node creation failed.")

                verdict_id = verdict_node_result['verdict_id']

                # 3. Link Claim to the NEW Verdict
                tx.run("""
                    MATCH (c), (v)
                    WHERE id(c) = $claim_node_id AND id(v) = $verdict_id
                    CREATE (c)-[:HAS_VERDICT]->(v)
                    """, claim_node_id=claim_node_id, verdict_id=verdict_id)


                # --- Evidence Snippet Handling: Remove old, create new ---
                # Delete existing HAS_EVIDENCE relationships and EvidenceSnippet nodes for this claim
                tx.run("""
                       MATCH (c:Claim)-[r:HAS_EVIDENCE]->(old_es:EvidenceSnippet)
                       WHERE id(c) = $claim_node_id
                       DELETE r, old_es
                       """, claim_node_id=claim_node_id)

                # 4. Create and link NEW EvidenceSnippet nodes
                for i, snippet_formatted_string in enumerate(top_rag_snippets):
                    source = "?" # Default source
                    content = snippet_formatted_string # Default content if parsing fails
                    try:
                        # Example format: "Snip 1: \"Snippet text...\" (Source URL)"
                        if '\" (' in snippet_formatted_string:
                             parts = snippet_formatted_string.split('\" (')
                             content_part = parts[0]
                             source = parts[1][:-1] if len(parts)>1 else "?" # Extract source, remove last ')'
                             if ': "' in content_part:
                                 content = content_part.split(': "', 1)[1]
                             else: content = content_part
                        else:
                             logging.warning(f"Could not parse snippet format: '{snippet_formatted_string}'")
                             content = snippet_formatted_string
                    except Exception as e:
                        logging.warning(f"Snippet parsing failed for: '{snippet_formatted_string}' - {e}")
                        content = snippet_formatted_string

                    # Create NEW EvidenceSnippet node
                    evidence_snippet_node_result = tx.run("""
                        CREATE (es:EvidenceSnippet {
                            content: $content,
                            source: $source,
                            snippet_index: $snippet_index
                        })
                        RETURN id(es) as snippet_id
                        """, content=content, source=source, snippet_index=i).single()

                    if evidence_snippet_node_result:
                        evidence_snippet_id = evidence_snippet_node_result['snippet_id']
                        # Link Claim to the NEW EvidenceSnippet
                        tx.run("""
                            MATCH (c), (es)
                            WHERE id(c) = $claim_node_id AND id(es) = $evidence_snippet_id
                            CREATE (c)-[:HAS_EVIDENCE]->(es)
                            """, claim_node_id=claim_node_id, evidence_snippet_id=evidence_snippet_id)
                    else:
                         logging.warning(f"Failed to create EvidenceSnippet node for snippet index {i}")


                # --- Entity Handling: MERGE entities and MERGE relationships ---
                # We MERGE entities to reuse them, and MERGE relationships to avoid duplicates.
                for entity in entities:
                    entity_text = entity.get('text')
                    entity_label = entity.get('label')
                    if not entity_text or not entity_label:
                        logging.warning(f"Skipping invalid entity data: {entity}")
                        continue

                    # Merge Entity node (create if not exists)
                    entity_node_result = tx.run("""
                        MERGE (e:Entity {text: $text, label: $label})
                        RETURN id(e) as entity_id
                        """, text=entity_text, label=entity_label).single()

                    if entity_node_result:
                        entity_id = entity_node_result['entity_id']
                        # MERGE the relationship between Claim and Entity
                        tx.run("""
                            MATCH (c), (e)
                            WHERE id(c) = $claim_node_id AND id(e) = $entity_id
                            MERGE (c)-[:MENTIONS]->(e)
                            """, claim_node_id=claim_node_id, entity_id=entity_id)
                    else:
                        logging.warning(f"Failed to merge Entity node for: {entity_text} ({entity_label})")


                tx.commit()
                logging.info(f"Stored/Updated claim '{claim}' in Neo4j (DB: {NEO4J_DATABASE}).")

            except Exception as e:
                if tx:
                    logging.error(f"Transaction failed for claim '{claim}'. Rolling back.")
                    try:
                        tx.rollback()
                    except Exception as rb_e:
                         logging.error(f"Error during rollback: {rb_e}")
                logging.error(f"Error storing/updating claim '{claim}' in Neo4j (DB: {NEO4J_DATABASE}): {e}", exc_info=True)

    # --- End UPDATED store_in_neo4j function ---

    def preprocess_and_filter(self, text: str) -> (list, list):
        if not text or not isinstance(text, str): logging.warning("Preprocess: Input empty."); return [], []
        sentences = []; checkable_claims_data = []; non_checkable_claims = []
        try:
            text=text.replace('.This','. This').replace('?This','? This').replace('!This','! This'); sentences=nltk.sent_tokenize(text); sentences=[s.strip() for s in sentences if s.strip()]; logging.info(f"Segmented {len(sentences)} sentences.")
        except Exception as e: logging.error(f"Tokenization error: {e}"); return [], []
        for sentence in sentences:
            if is_claim_checkable(sentence):
                sentence_ner = [];
                if self.nlp_available:
                    try: doc=NLP(sentence); sentence_ner=[{"text":ent.text,"label":ent.label_} for ent in doc.ents if ent.label_ in KG_RELEVANT_NER_LABELS]; logging.debug(f"NER for '{sentence}': {sentence_ner}")
                    except Exception as e: logging.error(f"NER fail '{sentence}': {e}")
                preprocessed=preprocess_claim_for_api(sentence); checkable_claims_data.append({"original_claim":sentence,"preprocessed_claim":preprocessed,"ner_entities":sentence_ner})
            else: non_checkable_claims.append(sentence)
        logging.info(f"Found {len(checkable_claims_data)} checkable claims.");
        if non_checkable_claims: logging.info(f"Filtered {len(non_checkable_claims)} non-checkable.")
        return checkable_claims_data, non_checkable_claims

    def classify_and_prioritize_claims(self, checkable_claims_data: list) -> list:
        if not checkable_claims_data: logging.warning("Prioritize: No data."); return []
        try:
            orig=[cd["original_claim"] for cd in checkable_claims_data]; embeds=self.embedding_model.encode(orig,convert_to_tensor=True,show_progress_bar=False)
            for i,cd in enumerate(checkable_claims_data):
                e=embeds[i].cpu().numpy(); s=0.5+(np.linalg.norm(e)/(np.linalg.norm(e)*25+1e-6)) if np.linalg.norm(e)>0 else 0.5; s=min(max(s,0.0),1.0); p=s
                # Add factual score directly to the dict here
                cd["factual_score"]=s;
                cd["priority"]=p; logging.info(f"Prioritize: '{cd['original_claim']}' -> Score:{s:.2f}, Prio:{p:.2f}")
            checkable_claims_data.sort(key=lambda x:x['priority'],reverse=True); logging.info(f"Prioritized {len(checkable_claims_data)} claims."); return checkable_claims_data
        except Exception as e: logging.error(f"Prioritization error: {e}",exc_info=True); return []

    def add_claims_to_queue(self, claims_to_process: list):
        if not claims_to_process: logging.warning("Queue: No claims."); return
        for cd_dict in claims_to_process: self.claim_queue.put(cd_dict)
        logging.info(f"Queued {len(claims_to_process)} claims for full processing. Queue Size: {self.claim_queue.qsize()}")

    def process_claim(self, claim_data_dict: dict):
        """Pipeline for claims *not* found in KG: GFactCheck -> GSearch -> RAG -> LLM -> Neo4j Store."""
        original_claim = claim_data_dict['original_claim']; preprocessed_claim = claim_data_dict['preprocessed_claim']
        ner_entities = claim_data_dict['ner_entities']; start_time = time.time()
        logging.info(f"Full Process: \"{original_claim}\" (API Query: \"{preprocessed_claim}\")")

        result = { # Initialize full result dict
            "original_claim": original_claim, "preprocessed_claim": preprocessed_claim, "ner_entities": ner_entities,
            "factual_score": claim_data_dict.get('factual_score', 0.0), # Get score from prioritization step
            "initial_verdict_raw": "N/A", "initial_evidence": "N/A", # Step 1 result
            "rag_status": "Not Attempted", "top_rag_snippets": [], # Step 2/3 results
            "final_verdict": "Pending", "final_explanation": "N/A", # Step 4 results (LLM)
            "source": "Full Pipeline" # Mark source
        }
        rag_evidence_for_llm = [] # Data for LLM

        # --- Step 1: Initial Fact Check ---
        try:
            initial_check_list = google_fact_check(preprocessed_claim, GOOGLE_API_KEY)
            self.raw_fact_checks[original_claim] = initial_check_list # Store raw
            initial_check = initial_check_list[0]
            result['initial_verdict_raw'] = initial_check.get('verdict', 'Error'); result['initial_evidence'] = initial_check.get('evidence', 'N/A')
            logging.info(f"GFactCheck Result: '{result['initial_verdict_raw']}'")
        except Exception as e: logging.error(f"GFactCheck fail: {e}"); result['initial_verdict_raw']="Error"; result['initial_evidence']=f"API fail:{e}"

        # --- Step 2: Google Custom Search (Best Effort) ---
        search_results = []; full_search_resp = {}
        try:
            logging.debug(f"Attempting GCustomSearch '{preprocessed_claim}'..."); full_search_resp, search_results = google_custom_search(preprocessed_claim, GOOGLE_API_KEY, GOOGLE_CSE_ID, NUM_SEARCH_RESULTS)
            self.raw_searches[original_claim] = {"query": preprocessed_claim, "response": full_search_resp, "results": search_results} # Store raw
            if not search_results and not full_search_resp: result["rag_status"] = "Search Failed (API Error)"
            elif not search_results: result["rag_status"] = "Search OK, No Results"
            else: result["rag_status"] = "Search OK, Results Found"
            logging.info(f"GCustomSearch status: {result['rag_status']}")
        except Exception as e: logging.error(f"GCustomSearch error: {e}"); result["rag_status"]="Search Failed (Code Error)"; search_results=[]

        # --- Step 3: RAG (Best Effort) ---
        if search_results:
            documents=[]; vector_store=None
            try:
                # Simple relevance check: preprocessed words in snippet
                sig_words={w.lower() for w in preprocessed_claim.split() if len(w)>3};
                documents=[Document(page_content=sr['snippet'],metadata={'source':sr['link'],'title':sr['title']}) for sr in search_results if any(w in sr['snippet'].lower() for w in sig_words)]

                if documents:
                    vector_store=FAISS.from_documents(documents,self.langchain_embeddings); logging.debug("FAISS index OK.")
                    retrieved_docs=vector_store.similarity_search(original_claim, k=RAG_K);
                    rag_evidence_for_llm=[{"content":doc.page_content,"metadata":doc.metadata} for doc in retrieved_docs]
                    # Store formatted snippets for Neo4j and display
                    result['top_rag_snippets']=[f"Snip {j+1}: \"{d['content'][:150].strip()}...\" ({d['metadata'].get('source','?')})" for j,d in enumerate(rag_evidence_for_llm)] # Slightly longer snippet preview
                    result["rag_status"] = f"RAG OK ({len(rag_evidence_for_llm)} snippets retrieved)"
                else:
                    result["rag_status"] = "Search OK, No Relevant Docs for Index"
                    result['top_rag_snippets']=[]
                    rag_evidence_for_llm=[]
            except Exception as e:
                logging.error(f"RAG fail: {e}",exc_info=False);
                result["rag_status"]=f"RAG Failed ({type(e).__name__})"
                rag_evidence_for_llm=[]
                result['top_rag_snippets']=[]
        else: # Handle case where search didn't yield results
             result["rag_status"] = result.get("rag_status", "Search Returned No Results") # Keep previous status if it was an error
             rag_evidence_for_llm = []
             result['top_rag_snippets'] = []

        # --- Step 4: LLM Final Verdict from RAG ---
        logging.info(f"LLM generating final verdict for '{original_claim}' from RAG...");
        llm_final_result = get_llm_final_verdict_from_rag(
            original_claim, # Base verdict on original claim
            rag_evidence_for_llm, # Use only RAG evidence
            result['rag_status']
        )
        result['final_verdict'] = f"{llm_final_result['final_label']} (Confidence: {llm_final_result['confidence']:.2f})"
        result['final_explanation'] = llm_final_result['explanation'] # LLM's justification based on RAG

        # --- Step 5: Store in Neo4j ---
        try:
            self.store_in_neo4j(result) # Call the updated function
        except Exception as neo4j_e:
            # Log error but continue processing other claims
            logging.error(f"Neo4j storage failed for claim '{original_claim}': {neo4j_e}", exc_info=True)


        processing_time = time.time() - start_time
        logging.info(f"Verdict (Full Pipeline) for '{original_claim}': {result['final_verdict']}. (Explain: {result['final_explanation']}). (Time:{processing_time:.2f}s)")
        with self.results_lock: self.results[original_claim] = result # Store final result keyed by original claim

    def worker(self):
        t_obj=current_thread(); t_name=t_obj.name; logging.info(f"W {t_name} start.")
        while True:
            cd_dict=None
            try:
                cd_dict=self.claim_queue.get(timeout=1)
                self.process_claim(cd_dict) # This is the full pipeline process
                self.claim_queue.task_done()
            except queue.Empty:
                logging.info(f"W {t_name} queue empty."); break
            except Exception as e:
                orig_claim=cd_dict.get('original_claim','?') if cd_dict else '?'; logging.error(f"W {t_name} error processing claim '{orig_claim}': {e}",exc_info=True)
                if cd_dict:
                    # Store error state in results
                    with self.results_lock:
                        self.results[orig_claim]={
                            "original_claim":orig_claim,
                            "preprocessed_claim": cd_dict.get('preprocessed_claim','?'),
                            "final_verdict":"Processing Error",
                            "final_explanation":f"Worker error: {e}",
                            "source": "Error" # Mark source as Error
                            }
                    try:
                        self.claim_queue.task_done() # Mark task done even on error
                    except ValueError: pass # If already marked done elsewhere?
        logging.info(f"W {t_name} finish.")

    def train_and_run_shap(self, claims_processed_fully: list):
        """Runs SHAP analysis only on claims that went through the full pipeline."""
        if not self.shap_available: logging.warning("SHAP unavailable."); self.shap_explanations = [{"claim":cd.get('original_claim', '?'),"shap_values":"[SHAP Unavailable]"} for cd in claims_processed_fully]; return
        if not claims_processed_fully: logging.info("SHAP: No claims went through the full pipeline, skipping SHAP."); self.shap_explanations = []; return

        logging.info(f"Attempting SHAP explanations for {len(claims_processed_fully)} fully processed claims...");
        valid_claims_data = [cd for cd in claims_processed_fully if cd.get("original_claim")]
        if not valid_claims_data: logging.warning("No valid claims data for SHAP."); self.shap_explanations=[]; return

        sentences = [cd['original_claim'] for cd in valid_claims_data]
        if not sentences: logging.warning("No sentences extracted for SHAP."); self.shap_explanations=[]; return

        # Initialize explanations structure
        self.shap_explanations=[{"claim":cd['original_claim'],"shap_values":"[SHAP Pending]"} for cd in valid_claims_data]

        try:
            embed_dim=self.embedding_model.get_sentence_embedding_dimension()
        except Exception:
            logging.warning("Could not get embedding dimension for SHAP fallback.")
            embed_dim = 384 # Fallback dimension

        try:
            embeddings=self.embedding_model.encode(sentences,convert_to_tensor=False,show_progress_bar=False); logging.debug(f"SHAP embeds: {embeddings.shape if isinstance(embeddings,np.ndarray) else 'Invalid Type'}")

            def predict_scores(np_embeds):
                 # Input validation for prediction function
                 if not isinstance(np_embeds, np.ndarray):
                     try:
                         np_embeds = np.array(np_embeds)
                         if np_embeds.ndim == 1: # Handle single instance prediction
                             np_embeds = np_embeds.reshape(1, -1)
                         assert np_embeds.ndim == 2
                     except Exception as pred_e:
                          logging.error(f"SHAP predict input error: {pred_e}. Input type: {type(np_embeds)}")
                          # Return default score for expected number of inputs
                          num_inputs = len(np_embeds) if hasattr(np_embeds, '__len__') else 1
                          return np.full(num_inputs, 0.5)

                 scs=[];
                 for e in np_embeds:
                     # Ensure e is treated as a 1D array for norm calculation
                     e_1d = e.flatten()
                     r=float(np.linalg.norm(e_1d));
                     s=0.5+(r/(r*25+1e-6)) if r>0 else 0.5;
                     scs.append(min(max(s,0.0),1.0))
                 return np.array(scs)

            # Validate embeddings shape for SHAP
            if not isinstance(embeddings, np.ndarray) or embeddings.ndim != 2 or embeddings.shape[0] < 1:
                logging.error(f"Invalid embeddings for SHAP: shape={embeddings.shape if isinstance(embeddings, np.ndarray) else type(embeddings)}. Need 2D numpy array.");
                self.shap_explanations=[{"claim":cd['original_claim'],"shap_values":"[SHAP Embed Err]"} for cd in valid_claims_data]; return

            logging.debug("Creating SHAP background data..."); bg_data=None
            try:
                 n_bg_samples = min(100, embeddings.shape[0]) # Limit background samples
                 n_clusters = min(10, n_bg_samples) # Clusters should be <= samples
                 if embeddings.shape[0] >= n_clusters and n_clusters > 1:
                     bg_obj = shap.kmeans(embeddings, n_clusters)
                     if hasattr(bg_obj, 'data') and isinstance(bg_obj.data, np.ndarray):
                         bg_data = bg_obj.data
                         logging.debug(f"Using KMeans background data, shape: {bg_data.shape}")
                     else:
                          logging.warning("KMeans object missing '.data', using raw embeddings for background.")
                          bg_data = embeddings[:n_bg_samples] # Use subset
                 else: # Not enough samples for kmeans or only 1 sample
                     logging.debug(f"Using raw embeddings for background (samples: {embeddings.shape[0]})")
                     bg_data = embeddings
            except Exception as ke:
                 logging.warning(f"SHAP KMeans failed: {ke}. Using raw embeddings for background.")
                 bg_data = embeddings[:min(100, embeddings.shape[0])] # Use subset on error too

            if bg_data is None or not isinstance(bg_data, np.ndarray):
                logging.error("SHAP background data preparation failed."); return

            # Ensure background data is 2D
            if bg_data.ndim == 1: bg_data = bg_data.reshape(1,-1)

            logging.debug("Initializing SHAP KernelExplainer...");
            explainer=shap.KernelExplainer(predict_scores, bg_data)

            logging.info(f"Calculating SHAP values for {embeddings.shape[0]} instance(s)...");
             # Use appropriate nsamples, potentially fewer for many instances
            n_samples_shap = min(50, 2 * bg_data.shape[1] + 2048) # SHAP default logic approximation
            shap_vals=explainer.shap_values(embeddings, nsamples=n_samples_shap)
            logging.info("SHAP values calculated.")

            calc_expl=[]
            # Handle different SHAP output shapes (scalar, 1D array, 2D array)
            if isinstance(shap_vals, (float, np.number)) and len(sentences) == 1: # Single scalar output
                 # Need to create a list matching embedding dim for consistency? Or store scalar?
                 # Storing a list of zeros might be misleading. Store specific message.
                 calc_expl.append({"claim":sentences[0],"shap_values":f"[SHAP Scalar Output: {shap_vals}]"})
                 logging.warning("SHAP returned a scalar value, expected array.")

            elif isinstance(shap_vals, np.ndarray):
                if shap_vals.ndim == 1 and len(sentences) == 1 and shap_vals.shape[0] == embed_dim: # Single instance, correct shape
                     calc_expl.append({"claim":sentences[0],"shap_values":shap_vals.tolist()})
                elif shap_vals.ndim == 2 and shap_vals.shape[0] == len(sentences) and shap_vals.shape[1] == embed_dim: # Multiple instances
                    for i, ct in enumerate(sentences): calc_expl.append({"claim":ct,"shap_values":shap_vals[i].tolist()})
                else: # Mismatched shape
                     logging.error(f"SHAP values shape mismatch. Got {shap_vals.shape}, expected ({len(sentences)}, {embed_dim}) or ({embed_dim},) for single instance.");
                     calc_expl = [{"claim":cd['original_claim'],"shap_values":f"[SHAP Calc Err Shape: {shap_vals.shape}]"} for cd in valid_claims_data]
            else: # Unexpected output type
                 logging.error(f"SHAP values unexpected type: {type(shap_vals)}.");
                 calc_expl = [{"claim":cd['original_claim'],"shap_values":f"[SHAP Calc Err Type: {type(shap_vals).__name__}]"} for cd in valid_claims_data]

            if calc_expl: logging.info(f"SHAP results stored for {len(calc_expl)} claims.")
            self.shap_explanations = calc_expl # Update with calculated explanations

        except Exception as e:
            logging.error(f"SHAP generation error: {e}",exc_info=True);
            # Provide fallback explanation based on error
            self.shap_explanations=[{"claim":cd['original_claim'],"shap_values":f"[SHAP Error: {type(e).__name__}]"} for cd in valid_claims_data]


    def generate_chain_of_thought(self, all_processed_claims: list, non_checkable_claims: list) -> str:
        """Generates CoT including KG hits and fully processed claims."""
        cot = ["Chain of Thought Summary:"]
        # 1. Preprocessing Summary
        cot.append("1. Input Segmentation & Filtering:")
        total_initial = len(all_processed_claims) + len(non_checkable_claims)
        cot.append(f"   - Initial Sentences: {total_initial}")
        if non_checkable_claims: cot.append(f"   - Filtered Non-Checkable ({len(non_checkable_claims)}): {non_checkable_claims}")
        cot.append(f"   - Checkable Claims Identified: {len(all_processed_claims)}")

        # 2. Preprocessing & Prioritization for Checkable Claims
        checkable_claims_str = [f"'{c.get('original_claim','?')}' (Prio:{c.get('priority',0):.2f})" for c in all_processed_claims if c.get('source') != 'Knowledge Graph'] # Show priority only if calculated
        if checkable_claims_str: cot.append(f"   - Preprocessed & Prioritized ({len(checkable_claims_str)}): [{', '.join(checkable_claims_str)}]")

        # 3. KG Check & Pipeline Execution Summary
        kg_hits = [c for c in all_processed_claims if c.get('source') == 'Knowledge Graph']
        pipeline_processed = [c for c in all_processed_claims if c.get('source') == 'Full Pipeline']
        errors = [c for c in all_processed_claims if c.get('source') == 'Error']

        cot.append("2. Knowledge Graph Check & Processing:")
        if kg_hits: cot.append(f"   - KG Hits ({len(kg_hits)}): Found existing reliable verdicts.")
        else: cot.append("   - KG Hits: None found matching criteria.")
        if pipeline_processed: cot.append(f"   - Full Pipeline ({len(pipeline_processed)}): Claims processed via APIs and LLM.")
        else: cot.append("   - Full Pipeline: No claims required full processing.")
        if errors: cot.append(f"   - Errors ({len(errors)}): Claims encountered processing errors.")

        # 4. Detailed Results Summary (combining KG and Pipeline)
        cot.append("3. Processed Claim Results:")
        with self.results_lock: # Access self.results safely if needed, though all_processed_claims has the final data
            for i, res in enumerate(all_processed_claims):
                claim = res.get('original_claim','?')
                source = res.get('source', '?')
                cot.append(f"   - Claim {i+1}: '{claim}' (Source: {source})")
                if source == "Knowledge Graph":
                    cot.append(f"     - Final Verdict: {res.get('final_verdict','?')}")
                    cot.append(f"     - Explanation: {res.get('final_explanation','?')}")
                elif source == "Full Pipeline":
                     cot.append(f"     - Initial Check: Verdict='{res.get('initial_verdict_raw','?')}'")
                     cot.append(f"     - RAG Status: {res.get('rag_status','?')}")
                     cot.append(f"     - LLM Final Verdict: {res.get('final_verdict','?')}")
                     cot.append(f"     - LLM Justification: {res.get('final_explanation','?')}")
                     if self.neo4j_driver: cot.append(f"     - Neo4j Storage: Stored/Updated")
                elif source == "Error":
                     cot.append(f"     - Final Verdict: {res.get('final_verdict','?')}")
                     cot.append(f"     - Explanation: {res.get('final_explanation','?')}")
                else: # Missing result? Should not happen with consolidation logic
                     cot.append(f"     - Status: Result Missing or Unknown Source!")

        # 5. SHAP Summary (based on pipeline_processed claims)
        cot.append("4. SHAP Analysis (for fully processed claims):")
        if self.shap_explanations:
             # Filter explanations for claims that were actually processed
             processed_claims_originals = {c['original_claim'] for c in pipeline_processed}
             relevant_explanations = [ex for ex in self.shap_explanations if ex.get('claim') in processed_claims_originals]

             if relevant_explanations:
                 sh_sum=[]; has_real=False
                 for ex in relevant_explanations:
                     v=ex.get('shap_values',[]); s="[Err/Unavail]"
                     if isinstance(v,list) and v:
                         if all(isinstance(x,(int,float)) for x in v): s=f"[...{len(v)} SHAP vals...]" if not any(isinstance(val, str) and val.startswith('[SHAP') for val in v) and not all(abs(y)<1e-9 for y in v) else "[SHAP Err/Fallback]"; has_real=True if s.startswith("[...") else has_real
                         else: s=str(v) # Show content if not numbers (e.g., error string in list)
                     elif isinstance(v,str): s=v # Error string like "[SHAP Embed Err]"
                     elif not v: s="[No Data]"
                     sh_sum.append(f"'{ex.get('claim','?')}': {s}")
                 status = "Generated." if has_real else "Failed/Unavailable."
                 cot.append(f"   - SHAP Status: {status} Details: {{{', '.join(sh_sum)}}}")
             else:
                 cot.append("   - SHAP analysis skipped (no relevant results for fully processed claims).")
        else:
            cot.append("   - SHAP analysis skipped or no results structure.")

        return "\n".join(cot)

    # --- MODIFIED check method ---
    def check(self, text: str, num_workers: int = 2) -> dict:
        start=time.time(); logging.info(f"Starting check: \"{text[:100]}...\"")
        # Reset results and intermediate data
        with self.results_lock:
            self.results={}; self.shap_explanations=[]; self.raw_fact_checks={}; self.raw_searches={}
        while not self.claim_queue.empty():
            try: self.claim_queue.get_nowait(); self.claim_queue.task_done()
            except queue.Empty: break
            except Exception: pass # Ignore errors clearing queue

        logging.info("Step 1: Preprocessing & Filtering...");
        checkable_claims_initial_data, non_checkable_sents=self.preprocess_and_filter(text)

        if not checkable_claims_initial_data:
            logging.warning("No checkable claims found after preprocessing.");
            return {"processed_claims":[], "non_checkable_claims":non_checkable_sents, "summary":"No checkable claims found.", "raw_fact_checks":{}, "raw_searches":{}}

        logging.info("Step 2: Prioritizing Checkable Claims...");
        prioritized_claims_data=self.classify_and_prioritize_claims(checkable_claims_initial_data)
        if not prioritized_claims_data:
             logging.warning("Prioritization failed."); # Should return empty list if input was empty
             # Proceed with empty list if prioritization itself failed? Or return error?
             # For now, proceed, assuming classify_and_prioritize handles internal errors.
             return {"processed_claims":[], "non_checkable_claims":non_checkable_sents, "summary":"Prioritization failed.", "raw_fact_checks":{}, "raw_searches":{}}


        logging.info("Step 3: Checking Knowledge Graph (Neo4j)...");
        claims_to_process_fully = []
        claims_found_in_kg = []
        for claim_data_dict in prioritized_claims_data:
             # Use preprocessed text for KG lookup
             preprocessed_claim_text = claim_data_dict.get('preprocessed_claim')
             if not preprocessed_claim_text:
                  logging.warning(f"Skipping KG check for claim with no preprocessed text: '{claim_data_dict.get('original_claim')}'")
                  claims_to_process_fully.append(claim_data_dict) # Process fully if preprocessing failed
                  continue

             kg_result = self.check_kg_for_claim(preprocessed_claim_text)

             if kg_result:
                 # Use the result found in the KG
                 claims_found_in_kg.append(kg_result)
                 # Add raw results from KG entry if needed? Not straightforward.
             else:
                 # Not found or error in KG, needs full processing
                 claims_to_process_fully.append(claim_data_dict)

        logging.info(f"KG Check Results: Found={len(claims_found_in_kg)}, Needs Full Processing={len(claims_to_process_fully)}")

        # --- Step 4: Process Claims needing Full Pipeline ---
        if claims_to_process_fully:
            logging.info("Step 4a: Queueing claims for full processing...");
            self.add_claims_to_queue(claims_to_process_fully)

            logging.info("Step 4b: Processing via Workers...");
            threads=[]; n_cpu=os.cpu_count() or 1;
            # Limit workers by available CPUs and queue size
            n_workers=min(num_workers, self.claim_queue.qsize(), n_cpu)
            if n_workers > 0:
                logging.info(f"Starting {n_workers} workers...");
                for i in range(n_workers): t=Thread(target=self.worker,name=f"Worker-{i+1}",daemon=True); t.start(); threads.append(t)
                self.claim_queue.join(); logging.info("Workers finished processing queue.")
                # Join threads explicitly after queue join to ensure completion
                for t in threads:
                    t.join(timeout=10.0) # Generous timeout
                    if t.is_alive(): logging.warning(f"Thread {t.name} still alive after join!")
            else:
                 logging.info("No claims needed full processing via workers.")
        else:
            logging.info("Step 4: Skipping full pipeline processing (all claims found in KG or none checkable).")

        # --- Step 5: Generate SHAP (only for fully processed claims) ---
        logging.info("Step 5: Generating SHAP (if installed)...");
        # Pass only the claims that *were* fully processed to SHAP
        self.train_and_run_shap(claims_to_process_fully) # SHAP runs on original data structure

        # --- Step 6: Consolidate Results ---
        logging.info("Step 6: Consolidating final results...");
        final_results_list = []
        processed_count = 0
        error_count = 0

        # Add results from KG directly
        final_results_list.extend(claims_found_in_kg)
        processed_count += len(claims_found_in_kg)

        # Add results from the full pipeline processing (from self.results)
        with self.results_lock:
            for cd in claims_to_process_fully: # Iterate through claims sent to workers
                 original_claim = cd['original_claim']
                 pipeline_result = self.results.get(original_claim)
                 if pipeline_result:
                      final_results_list.append(pipeline_result)
                      processed_count += 1
                      if pipeline_result.get("source") == "Error" or "Error" in pipeline_result.get("final_verdict",""):
                           error_count += 1
                 else:
                      # This should ideally not happen if workers store results/errors
                      logging.error(f"Result missing from self.results for fully processed claim: '{original_claim}'. Adding error entry.")
                      final_results_list.append({
                          "original_claim": original_claim,
                          "preprocessed_claim": cd.get('preprocessed_claim','?'),
                          "final_verdict": "Missing Result",
                          "final_explanation": "Result lost after worker processing.",
                          "source": "Error"
                          })
                      error_count += 1

        # Sort final list? Maybe by original order or priority? For now, KG first then pipeline.
        # Re-sort by original text order might be good for consistency? Or keep KG first?
        # Let's keep KG first, then pipeline results (order might be worker-dependent)

        # --- Step 7: Generate Summary ---
        # Pass the combined list to generate the summary
        summary = self.generate_chain_of_thought(final_results_list, non_checkable_sents)

        duration = time.time() - start
        logging.info(f"Check complete. Total checkable claims={len(prioritized_claims_data)}, KG Hits={len(claims_found_in_kg)}, Fully Processed={processed_count - len(claims_found_in_kg)}, Errors={error_count} in {duration:.2f}s.")

        # Return all relevant data structures
        return {
            "processed_claims": final_results_list, # Combined results
            "non_checkable_claims": non_checkable_sents,
            "summary": summary,
            "raw_fact_checks": self.raw_fact_checks, # Only contains results for fully processed claims
            "raw_searches": self.raw_searches,       # Only contains results for fully processed claims
            "shap_explanations": self.shap_explanations # Only contains results for fully processed claims
            }

    def close_neo4j(self):
        """Closes the Neo4j driver connection."""
        if self.neo4j_driver:
            try:
                self.neo4j_driver.close()
                logging.info("Neo4j driver closed.")
            except Exception as e:
                logging.error(f"Error closing Neo4j driver: {e}")


# --- End FactChecker Class ---

# --- Main Execution Block ---
if __name__ == "__main__":
    # Dependency Checks
    if shap is None: print("\nWARN: SHAP lib missing. pip install shap\n")
    else: print("SHAP library found.")
    if spacy is None: print("\nWARN: spaCy lib missing. pip install spacy\n")
    elif NLP is None: print("\nWARN: spaCy model 'en_core_web_sm' missing. python -m spacy download en_core_web_sm\n")
    else: print("spaCy library and model found.")
    if Groq is None: print(f"\nWARN: Groq lib missing. LLM ({LLM_PROVIDER}) disabled. pip install groq\n")
    elif groq_client is None: print(f"\nWARN: Groq client init failed (check key). LLM ({LLM_PROVIDER}) disabled.\n")
    else: print(f"{LLM_PROVIDER} client available (Model: {LLM_MODEL_NAME}).")

    print("Fact Checker Initializing...")
    if not GOOGLE_API_KEY or not GOOGLE_CSE_ID: logging.critical("CRIT: Google keys missing."); print("\nCRIT ERR: Google keys missing."); exit(1)
    else: logging.info("Google keys loaded.")
    if not NEO4J_PASSWORD: logging.critical("CRIT: NEO4J_PASSWORD missing."); print("\nCRIT ERR: Neo4j password missing."); exit(1)

    checker = None # Initialize checker outside try
    try:
        checker = FactChecker()
        # Initialization includes Neo4j connection test now
        if checker.neo4j_driver is None:
            print("\nCRIT ERR: Neo4j driver failed to initialize. Check logs. Exiting.")
            # Logs should contain the specific connection/auth error
            exit(1)
        print("Fact Checker Initialized Successfully.")
    except RuntimeError as e: # Model loading error
        logging.critical(f"Init fail (Model Load?): {e}",exc_info=True); print(f"\nCRIT ERR: Init failed. Logs:{log_file}. Err:{e}"); exit(1)
    except Exception as e: # Other unexpected init errors
        logging.critical(f"Unexpected init err: {e}",exc_info=True); print(f"\nCRIT UNEXPECTED init err. Logs:{log_file}. Err:{e}"); exit(1)

    # --- Example Input ---
    # Run 1: Will likely process most claims fully
    # Run 2: Should find several claims in the KG if Run 1 was successful
    input_text = (
        "COVID-19 vaccines are ineffective"    )
    print(f"\nInput Text:\n{input_text}\n")


    print("\n--- Starting Fact Check Pipeline ---\n")
    # Ensure checker is initialized before calling check
    if not checker:
         print("Checker not initialized. Exiting.")
         exit(1)

    results_data = checker.check(input_text, num_workers=2) # checker.check now returns the final dict

    # --- OUTPUT ORDER ---

    # 1. Raw Fact Check API Results (Only for claims processed fully)
    print("\n" + "="*25 + " Intermediate Output 1: Raw Google Fact Check API Results " + "="*15)
    if results_data.get("raw_fact_checks"):
        if results_data["raw_fact_checks"]:
            print(" (Note: Shows results only for claims requiring full API processing)")
            for claim, api_res_list in results_data["raw_fact_checks"].items():
                print(f"\nClaim (Original): \"{claim}\"")
                if api_res_list:
                    for res_item in api_res_list: print(f"  - Verdict: {res_item.get('verdict','?')} | Evidence: {res_item.get('evidence','?')}")
                else: print("  - No result stored (API call likely failed).")
        else: print("  - No Fact Check API calls were made or stored (or all claims were KG hits).")
    else: print("  - Raw Fact Check data structure missing.")
    print("="*81)

    # 2. Raw Google Custom Search Snippets (Optional, only for claims processed fully)
    SHOW_RAW_SEARCH = False # Keep this False unless debugging search itself
    if SHOW_RAW_SEARCH and results_data.get("raw_searches"):
        print("\n" + "="*25 + " Intermediate Output 2: Raw Google Custom Search Snippets " + "="*14)
        if results_data["raw_searches"]:
            print(" (Note: Shows results only for claims requiring full API processing)")
            for claim, search_data in results_data["raw_searches"].items():
                print(f"\nClaim (Original): \"{claim}\""); print(f"  (API Query: \"{search_data.get('query', '?')}\")")
                search_results = search_data.get("results", [])
                if search_results:
                    for i, item in enumerate(search_results): print(f"  {i+1}. T: {item.get('title','?')}\n     S: {item.get('snippet','?')}\n     L: {item.get('link','?')}")
                elif search_data.get("response") is not None: print("  - Search OK, no items.")
                else: print("  - Search API call likely failed.")
        else: print("  - No Custom Search API calls were made or stored (or all claims were KG hits).")
        print("="*81)

    # 3. Filtered Non-Checkable Claims
    print("\n" + "="*25 + " Preprocessing Output: Filtered Non-Checkable Sentences " + "="*16)
    if results_data.get("non_checkable_claims"):
        for i, claim in enumerate(results_data["non_checkable_claims"]): print(f"  {i+1}. \"{claim}\"")
    else: print("  - No sentences were filtered out.")
    print("="*81)

    # 4. Detailed Processed Claim Results (Includes KG hits and Fully Processed)
    print("\n" + "="*30 + " Final Processed Claim Details " + "="*30)
    if results_data and results_data.get("processed_claims"):
        # Sort results for consistent display (e.g., by original claim text)
        sorted_results = sorted(results_data["processed_claims"], key=lambda x: x.get('original_claim', ''))
        for i, res in enumerate(sorted_results):
            source = res.get('source', '?')
            print(f"\nClaim {i+1} (Original): \"{res.get('original_claim', '?')}\" [Source: {source}]")
            print(f"  - Preprocessed: \"{res.get('preprocessed_claim', 'N/A')}\"") # Show preprocessed text

            if source == "Full Pipeline":
                ner_ents = res.get('ner_entities', [])
                if ner_ents: print("  - NER Entities: {}".format(', '.join(["{}({})".format(e['text'], e['label']) for e in ner_ents])))
                else: print("  - NER Entities: None Found")
                print(f"  - Factual Score (0-1): {res.get('factual_score'):.2f}" if res.get('factual_score') is not None else "N/A")
                print(f"  - Initial Check Result: '{res.get('initial_verdict_raw','?')}'")
                print(f"  - RAG Status: {res.get('rag_status', '?')}")
                if res.get('top_rag_snippets'): print("  - Top RAG Snippets:"); [print(f"    {j+1}. {snip}") for j,snip in enumerate(res.get('top_rag_snippets',[]))]
                else: print("  - Top RAG Snippets: None")
                print(f"  - Final Verdict (RAG+LLM): {res.get('final_verdict', '?')}")
                print(f"  - LLM Justification: {res.get('final_explanation', '?')}")
                if checker and checker.neo4j_driver: print(f"  - Neo4j Storage: Stored/Updated in DB '{NEO4J_DATABASE}'")

            elif source == "Knowledge Graph":
                print(f"  - Final Verdict (From KG): {res.get('final_verdict', '?')}")
                print(f"  - KG Explanation: {res.get('final_explanation', '?')}")
                if 'kg_timestamp' in res: print(f"  - KG Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(res['kg_timestamp']))}" if res['kg_timestamp'] else "N/A")

            elif source == "Error":
                print(f"  - Final Verdict: {res.get('final_verdict', 'Error')}")
                print(f"  - Explanation: {res.get('final_explanation', 'N/A')}")
            else:
                 print("  - Unknown processing source or result format.")

    else: print("No checkable claims were processed or results available.")
    print("="*83)

    # 5. XAI (SHAP) Summary (Only for claims processed fully)
    print("\n" + "="*30 + " XAI (SHAP) Summary " + "="*37)
    if results_data.get("shap_explanations"):
         # Filter explanations based on claims that actually had SHAP run
         fully_processed_claims = {p['original_claim'] for p in results_data.get("processed_claims", []) if p.get('source') == 'Full Pipeline'}
         relevant_explanations = [ex for ex in results_data["shap_explanations"] if ex.get('claim') in fully_processed_claims]

         if relevant_explanations:
             print(f" (Note: Shows results only for {len(relevant_explanations)} claim(s) requiring full API processing)")
             shap_summary=[]; has_real=False
             for expl in relevant_explanations:
                 v=expl.get('shap_values',[]); s="[Err/Unavail]"
                 if isinstance(v,list) and v:
                     # Check if list contains actual numbers vs error strings/placeholders
                     is_numeric = all(isinstance(x,(int,float)) for x in v)
                     is_error_fallback = any(isinstance(val, str) and val.startswith('[SHAP') for val in v)

                     if is_numeric and not is_error_fallback:
                         s=f"[...{len(v)} values]" if not all(abs(y)<1e-9 for y in v) else "[Zero Values]"
                         has_real=True if s.startswith("[...") else has_real
                     elif is_error_fallback:
                          s = v[0] if v else "[SHAP Err Fallback]" # Display first error msg
                     else: s=str(v) # Display list content if not numbers/errors
                 elif isinstance(v,str): s=v # Handle top-level error strings
                 elif not v: s="[No Data]"
                 shap_summary.append(f"'{expl.get('claim','?')}': {s}")
             status = "Generated." if has_real else "Failed/Unavailable."
             print(f"  - SHAP Status: {status}")
             if shap_summary: print(f"  - Details: {{{', '.join(shap_summary)}}}")
             # Add reminder to check log file for errors
             if not has_real and shap is not None and relevant_explanations:
                  if any("[SHAP Error" in str(expl.get("shap_values","")) or "[SHAP Embed Err]" in str(expl.get("shap_values","")) for expl in relevant_explanations):
                       print(f"\n  *** SHAP Error Detected: Check '{log_file}' for traceback. ***")
         else:
              print("  - SHAP analysis not applicable (no claims required full processing or SHAP failed).")
    else: print("  - SHAP analysis skipped or no results structure.")
    print("="*86)

    # 6. Chain of Thought Summary
    print("\n" + "="*30 + " Chain of Thought Summary " + "="*30)
    print(results_data.get("summary", "No summary generated."))
    print("="*86)
    print(f"\nLog file generated at: {log_file}")

    # Close Neo4j connection cleanly
    if checker:
        checker.close_neo4j()

    print("\nScript finished.")