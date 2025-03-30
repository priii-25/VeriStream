# -*- coding: utf-8 -*-
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
    Groq = None; APIConnectionError=Exception; RateLimitError=Exception; APIStatusError=Exception # Dummy errors
try:
    import spacy
    try:
        NLP = spacy.load("en_core_web_sm"); logging.info("spaCy model loaded.")
    except OSError: logging.error("spaCy model not found."); NLP = None
except ImportError: spacy = None; NLP = None
# --- NEW: Import Transformers for Zero-Shot Classification ---
try:
    from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
    # Choose a zero-shot model
    # ZS_MODEL_NAME = "facebook/bart-large-mnli"
    ZS_MODEL_NAME = "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli" # Good multilingual alternative
    # Initialize the pipeline globally (can take time on first run)
    try:
        zs_classifier = pipeline("zero-shot-classification", model=ZS_MODEL_NAME, device=DEVICE) # Use same device as embeddings
        logging.info(f"Zero-shot classifier pipeline loaded ({ZS_MODEL_NAME}).")
    except Exception as e:
        logging.error(f"Failed to load zero-shot classifier pipeline '{ZS_MODEL_NAME}': {e}")
        zs_classifier = None
except ImportError:
    logging.warning("Hugging Face `transformers` library not found. pip install transformers torch (or tensorflow/flax). Falling back to basic claim filtering.")
    zs_classifier = None
# --- End Imports ---

import queue
from threading import Thread, Lock, current_thread
import sys
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS

# --- Configuration (Mostly same) ---
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

# Clients
groq_client = None
# (Groq client init same as before)
if Groq and GROQ_API_KEY:
    try: groq_client = Groq(api_key=GROQ_API_KEY); logging.info("Groq client initialized.")
    except Exception as e: logging.error(f"Failed Groq client init: {e}")
elif not Groq: logging.warning("Groq library not installed. LLM disabled.")
elif not GROQ_API_KEY: logging.warning("GROQ_API_KEY not found. LLM disabled.")

# Models & Parameters
EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2'
DEVICE = 'cpu' # Ensure DEVICE is set correctly for transformers pipeline too
LLM_PROVIDER = "Groq"
LLM_MODEL_NAME = "llama3-8b-8192"
NUM_SEARCH_RESULTS = 3
RAG_K = 3
# --- NEW: Zero-Shot Classifier Settings ---
ZS_CANDIDATE_LABELS = ["factual claim", "opinion or subjective statement", "question", "command or instruction", "incomplete phrase", "other non-factual"]
ZS_FACTUAL_LABEL = "factual claim"
ZS_CONFIDENCE_THRESHOLD = 0.60 # Minimum confidence score to classify as factual
# --- End Zero-Shot Settings ---

KG_RELEVANT_NER_LABELS = {"PERSON", "ORG", "GPE", "LOC", "DATE", "TIME", "MONEY", "QUANTITY", "PERCENT", "CARDINAL", "ORDINAL", "PRODUCT", "EVENT", "WORK_OF_ART", "LAW", "NORP", "FAC"}

# Logging (same)
log_file = 'fact_checker.log'
with open(log_file, 'w', encoding='utf-8') as f: pass
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(threadName)s] - %(message)s', handlers=[logging.FileHandler(log_file, encoding='utf-8'), logging.StreamHandler(stream=sys.stdout)])

# NLTK (same)
try: nltk.data.find('tokenizers/punkt')
except: logging.info("Downloading NLTK 'punkt'..."); nltk.download('punkt', quiet=True); logging.info("'punkt' downloaded.")

# --- API Functions (Unchanged) ---
def google_fact_check(query: str, api_key: str) -> list:
    if not api_key: logging.error("GFactCheck Key missing."); return [{'verdict': 'Error', 'evidence': 'API Key missing'}]
    logging.debug(f"Querying GFactCheck API with: '{query}'")
    try: url="https://factchecktools.googleapis.com/v1alpha1/claims:search"; params={'query': query,'key': api_key,'languageCode': 'en'}; response=requests.get(url, params=params, timeout=10); response.raise_for_status(); data=response.json()
    except requests.exceptions.Timeout: logging.error(f"Timeout GFactCheck: '{query}'"); return [{'verdict': 'Error', 'evidence': 'API timeout'}]
    except requests.exceptions.RequestException as e: logging.error(f"Error GFactCheck API: {e}"); return [{'verdict': 'Error', 'evidence': f'API failed: {e}'}]
    except Exception as e: logging.error(f"Error processing GFactCheck: {e}"); return [{'verdict': 'Error', 'evidence': f'Resp processing fail: {e}'}]
    if 'claims' in data and data['claims']:
        claim = data['claims'][0]
        if claim.get('claimReview'): rev=claim['claimReview'][0]; v=rev.get('textualRating','?'); e=f"{rev.get('publisher',{}).get('name','?')} - {rev.get('url','?')}"; return [{'verdict':v,'evidence':e}]
    return [{'verdict': 'Unknown', 'evidence': 'No matching fact-check'}]

def google_custom_search(query: str, api_key: str, cse_id: str, num: int) -> (dict, list):
    if not api_key or not cse_id: logging.error("GCustomSearch keys missing."); return {}, []
    logging.debug(f"Querying GCustomSearch API with: '{query}'")
    try: url="https://www.googleapis.com/customsearch/v1"; params={'key':api_key,'cx':cse_id,'q':query,'num':num}; response=requests.get(url, params=params, timeout=15); response.raise_for_status(); full_resp=response.json()
    except requests.exceptions.Timeout: logging.error(f"Timeout GCustomSearch: '{query}'"); return {}, []
    except requests.exceptions.RequestException as e: logging.error(f"Error GCustomSearch API: {e}"); return {}, []
    except Exception as e: logging.error(f"Error processing GCustomSearch: {e}"); return {}, []
    items=full_resp.get('items',[]); results=[{'title':i.get('title','?'),'snippet':i.get('snippet','').replace('\n',' '),'link':i.get('link','')} for i in items if i.get('snippet')]
    logging.info(f"GCustomSearch got {len(results)} results for query '{query}'."); return full_resp, results

# --- LLM Function (Unchanged) ---
def get_llm_final_verdict_from_rag(claim: str, rag_evidence: list, rag_status_msg: str) -> dict:
    if not groq_client: return {"final_label":"LLM Error","explanation":"Groq client missing.","confidence":0.1}
    rag_fmt=f"\nRetrieved Snippets (RAG Status:{rag_status_msg}):\n";
    if rag_evidence:
        for i,d in enumerate(rag_evidence): rag_fmt+=f"{i+1}. Snip:\"{d.get('content','?')}\"\n   Src:{d.get('metadata',{}).get('source','?')}\n"
    else: rag_fmt+="None Provided/Failed.\n"
    prompt=f"""Strictly evaluate the 'Claim' based *only* on the 'Retrieved Snippets'. Ignore external knowledge.

Claim: "{claim}"
{rag_fmt}
Task: Is the claim True or False based *only* on the snippets? If snippets are missing, failed, irrelevant, or insufficient, verdict *must* be "False". Output *only* a valid JSON object with keys "verdict" (string "True" or "False") and "justification" (string, 1-2 sentences citing snippets/lack thereof).

Example: {{"verdict": "False", "justification": "RAG failed, claim unverified."}}
Your JSON Response:
"""
    logging.debug(f"LLM Final Verdict Prompt for '{claim}':\n{prompt}")
    try:
        response=groq_client.chat.completions.create(model=LLM_MODEL_NAME,messages=[{"role":"system","content":"Output valid JSON."}, {"role":"user","content":prompt}], temperature=0.0,max_tokens=150)
        llm_txt=response.choices[0].message.content; logging.debug(f"Groq LLM Raw: {llm_txt}")
        try: # Parse
            if llm_txt.strip().startswith("```json"): llm_txt=llm_txt.strip()[7:-3].strip()
            elif llm_txt.strip().startswith("```"): llm_txt=llm_txt.strip()[3:-3].strip()
            res=json.loads(llm_txt); v=res.get("verdict"); j=res.get("justification")
            if isinstance(v,str) and v in ["True","False"] and isinstance(j,str) and j: conf=0.90 if v=="True" else (0.88 if rag_evidence else 0.85); return {"final_label":v,"explanation":j,"confidence":conf}
            else: logging.error(f"LLM invalid content: {llm_txt}"); return {"final_label":"LLM Error","explanation":f"Invalid JSON content: {llm_txt}","confidence":0.1}
        except Exception as e: logging.error(f"LLM parse fail: {e}. Resp:{llm_txt}"); return {"final_label":"LLM Error","explanation":f"LLM JSON parse err: {llm_txt}","confidence":0.1}
    except APIConnectionError as e: logging.error(f"Groq ConnErr: {e}"); return {"final_label":"LLM Error","explanation":f"API Conn Err","confidence":0.1}
    except RateLimitError as e: logging.error(f"Groq RateLimit: {e}"); time.sleep(5); return {"final_label":"LLM Error","explanation":f"API Rate Limit","confidence":0.1}
    except APIStatusError as e: logging.error(f"Groq Status {e.status_code}"); return {"final_label":"LLM Error","explanation":f"API Status {e.status_code}","confidence":0.1}
    except Exception as e: logging.error(f"Unexpected Groq API err: {e}",exc_info=True); return {"final_label":"LLM Error","explanation":f"API Err","confidence":0.1}

# --- NEW: Zero-Shot Claim Checkability Function ---
def is_claim_checkable_zero_shot(sentence: str) -> bool:
    """Uses a zero-shot classification model to determine if a sentence is a factual claim."""
    if not zs_classifier:
        logging.warning("Zero-shot classifier not available, falling back to basic heuristics.")
        # Call the old heuristic function as a fallback
        from fact_checker_helpers import is_claim_checkable_heuristic # Assuming heuristics moved to a helper file
        return is_claim_checkable_heuristic(sentence)

    if not sentence or not isinstance(sentence, str): return False

    try:
        # Classify the sentence using the predefined labels
        result = zs_classifier(sentence, ZS_CANDIDATE_LABELS, multi_label=False) # Assuming single best label

        top_label = result['labels'][0]
        top_score = result['scores'][0]

        logging.debug(f"ZeroShot Check: '{sentence}' -> Top Label: {top_label} (Score: {top_score:.3f})")

        # Check if the top label is the desired "factual claim" label and meets the confidence threshold
        is_factual = (top_label == ZS_FACTUAL_LABEL and top_score >= ZS_CONFIDENCE_THRESHOLD)

        if not is_factual:
            logging.info(f"Claim filtered (ZeroShot): '{sentence}' -> Classified as: {top_label} (Score: {top_score:.3f})")
        return is_factual

    except Exception as e:
        logging.error(f"Error during zero-shot classification for '{sentence}': {e}")
        # Fallback strategy on error: maybe default to 'True' (checkable) or use heuristics?
        # Let's default to checkable on error to avoid missing claims, but log it.
        logging.warning(f"Zero-shot classification failed for '{sentence}', assuming checkable as fallback.")
        return True
# --- End Zero-Shot Function ---

# --- Helper Functions ---
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
                 while True:
                      potential_heads = [c for c in obj_or_comp.children if c.pos_ in ("NOUN", "PROPN", "ADJ", "NUM")]
                      if potential_heads: obj_or_comp = potential_heads[0]; continue
                      else: break
        if subj: simplified_parts.append(subj.text)
        if negation: simplified_parts.append("not")
        aux_verb = next((t for t in root.children if t.dep_=='aux' or (t.dep_=='attr'and t.pos_=='AUX')), None)
        if root.pos_!='NOUN' and root.lemma_ not in simplified_parts: simplified_parts.append(root.lemma_)
        elif root.head.pos_=='VERB' and root.head.lemma_ not in simplified_parts: simplified_parts.append(root.head.lemma_)
        elif aux_verb and aux_verb.lemma_ not in simplified_parts: simplified_parts.append(aux_verb.lemma_)
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
        if NLP is None: logging.warning("spaCy model not loaded. NER/Preprocessing basic.")
        self.nlp_available = NLP is not None
        if zs_classifier is None: logging.warning("Zero-shot classifier not loaded. Claim filtering basic.")
        self.zs_classifier_available = zs_classifier is not None

        self.claim_queue=queue.Queue(); self.results={}; self.results_lock=Lock(); self.shap_explanations=[]
        self.raw_fact_checks={}; self.raw_searches={};

    def preprocess_and_filter(self, text: str) -> (list, list):
        """Segments, filters (using zero-shot if available), NER, preprocesses."""
        if not text or not isinstance(text, str): logging.warning("Preprocess: Input empty."); return [], []
        sentences = []; checkable_claims_data = []; non_checkable_claims = []
        try:
            text=text.replace('.This','. This').replace('?This','? This').replace('!This','! This'); sentences=nltk.sent_tokenize(text); sentences=[s.strip() for s in sentences if s.strip()]; logging.info(f"Segmented {len(sentences)} sentences.")
        except Exception as e: logging.error(f"Tokenization error: {e}"); return [], []

        for sentence in sentences:
            # --- Use Zero-Shot Classifier for Filtering ---
            is_checkable = is_claim_checkable_zero_shot(sentence)
            # ---------------------------------------------
            if is_checkable:
                sentence_ner = [];
                if self.nlp_available:
                    try: doc=NLP(sentence); sentence_ner=[{"text":ent.text,"label":ent.label_} for ent in doc.ents if ent.label_ in KG_RELEVANT_NER_LABELS]; logging.debug(f"NER for '{sentence}': {sentence_ner}")
                    except Exception as e: logging.error(f"NER fail '{sentence}': {e}")
                preprocessed=preprocess_claim_for_api(sentence); checkable_claims_data.append({"original_claim":sentence,"preprocessed_claim":preprocessed,"ner_entities":sentence_ner})
            else:
                non_checkable_claims.append(sentence)
        logging.info(f"Found {len(checkable_claims_data)} checkable claims after filtering.")
        if non_checkable_claims: logging.info(f"Filtered {len(non_checkable_claims)} non-checkable sentences.")
        return checkable_claims_data, non_checkable_claims

    def classify_and_prioritize_claims(self, checkable_claims_data: list) -> list:
        # (Priority scoring remains the same)
        if not checkable_claims_data: logging.warning("Prioritize: No data."); return []
        try:
            orig=[cd["original_claim"] for cd in checkable_claims_data]; embeds=self.embedding_model.encode(orig,convert_to_tensor=True,show_progress_bar=False)
            for i,cd in enumerate(checkable_claims_data):
                e=embeds[i].cpu().numpy(); s=0.5+(np.linalg.norm(e)/(np.linalg.norm(e)*25+1e-6)) if np.linalg.norm(e)>0 else 0.5; s=min(max(s,0.0),1.0); p=s
                cd["score"]=s; cd["priority"]=p; logging.info(f"Prioritize: '{cd['original_claim']}' -> Prio:{p:.2f}")
            checkable_claims_data.sort(key=lambda x:x['priority'],reverse=True); logging.info(f"Prioritized {len(checkable_claims_data)} claims."); return checkable_claims_data
        except Exception as e: logging.error(f"Prioritization error: {e}",exc_info=True); return []

    def add_claims_to_queue(self, claims_to_process: list):
        # (Queueing remains the same)
        if not claims_to_process: logging.warning("Queue: No claims."); return
        for cd_dict in claims_to_process: self.claim_queue.put(cd_dict)
        logging.info(f"Queued {len(claims_to_process)} claims. Size: {self.claim_queue.qsize()}")

    def process_claim(self, claim_data_dict: dict):
        """Pipeline: GFactCheck(prep) -> GCustomSearch(prep) -> RAG(orig) -> LLM Verdict(orig, RAG)."""
        original_claim = claim_data_dict['original_claim']; preprocessed_claim = claim_data_dict['preprocessed_claim']
        ner_entities = claim_data_dict['ner_entities']; start_time = time.time()
        logging.info(f"Processing: \"{original_claim}\" (API Query: \"{preprocessed_claim}\")")
        result = { # Initialize full result dict
            "original_claim": original_claim, "preprocessed_claim": preprocessed_claim, "ner_entities": ner_entities,
            "factual_score": claim_data_dict.get('score', None),
            "initial_verdict_raw": "N/A", "initial_evidence": "N/A", # Step 1 result
            "rag_status": "Not Attempted", "top_rag_snippets": [], # Step 2/3 results
            "final_verdict": "Pending", "final_explanation": "N/A" # Step 4 results (LLM)
        }; rag_evidence_for_llm = []

        # --- Step 1: Initial Fact Check ---
        try:
            initial_check_list = google_fact_check(preprocessed_claim, GOOGLE_API_KEY)
            self.raw_fact_checks[original_claim] = initial_check_list
            initial_check = initial_check_list[0]
            result['initial_verdict_raw'] = initial_check.get('verdict', 'Error'); result['initial_evidence'] = initial_check.get('evidence', 'N/A')
            logging.info(f"GFactCheck Result: '{result['initial_verdict_raw']}'")
        except Exception as e: logging.error(f"GFactCheck fail: {e}"); result['initial_verdict_raw']="Error"; result['initial_evidence']=f"API fail:{e}"

        # --- Step 2: Google Custom Search (Best Effort) ---
        search_results = []; full_search_resp = {}
        try:
            logging.debug(f"Attempting GCustomSearch '{preprocessed_claim}'..."); full_search_resp, search_results = google_custom_search(preprocessed_claim, GOOGLE_API_KEY, GOOGLE_CSE_ID, NUM_SEARCH_RESULTS)
            self.raw_searches[original_claim] = {"query": preprocessed_claim, "response": full_search_resp, "results": search_results}
            if not search_results and not full_search_resp: result["rag_status"] = "Search Failed (API Error)"
            elif not search_results: result["rag_status"] = "Search OK, No Results"
            else: result["rag_status"] = "Search OK, Results Found"
            logging.info(f"GCustomSearch status: {result['rag_status']}")
        except Exception as e: logging.error(f"GCustomSearch error: {e}"); result["rag_status"]="Search Failed (Code Error)"; search_results=[]

        # --- Step 3: RAG (Best Effort) ---
        if search_results:
            documents=[]; vector_store=None
            try:
                sig_words={w.lower() for w in preprocessed_claim.split() if len(w)>3}; documents=[Document(page_content=sr['snippet'],metadata={'source':sr['link'],'title':sr['title']}) for sr in search_results if any(w in sr['snippet'].lower() for w in sig_words)]
                if documents:
                    vector_store=FAISS.from_documents(documents,self.langchain_embeddings); logging.debug("FAISS index OK.")
                    retrieved_docs=vector_store.similarity_search(original_claim, k=RAG_K); rag_evidence_for_llm=[{"content":doc.page_content,"metadata":doc.metadata} for doc in retrieved_docs]
                    result['top_rag_snippets']=[f"Snip {j+1}: \"{d['content'][:100]}...\" ({d['metadata'].get('source','?')})" for j,d in enumerate(rag_evidence_for_llm)]
                    result["rag_status"] = f"RAG OK ({len(rag_evidence_for_llm)} snippets)"
                else: result["rag_status"] = "Search OK, No Relevant Docs for Index"
            except Exception as e: logging.error(f"RAG fail: {e}",exc_info=False); result["rag_status"]=f"RAG Failed ({type(e).__name__})"; rag_evidence_for_llm=[]; result['top_rag_snippets']=[]

        # --- Step 4: LLM Final Verdict from RAG ---
        logging.info(f"LLM generating final verdict for '{original_claim}' from RAG...");
        llm_final_result = get_llm_final_verdict_from_rag(original_claim, rag_evidence_for_llm, result['rag_status'])
        result['final_verdict'] = f"{llm_final_result['final_label']} (Confidence: {llm_final_result['confidence']:.2f})"
        result['final_explanation'] = llm_final_result['explanation']

        processing_time = time.time() - start_time
        logging.info(f"Final Verdict for '{original_claim}': {result['final_verdict']}. (RAG-LLM Explain: {result['final_explanation']}). (Time:{processing_time:.2f}s)")
        with self.results_lock: self.results[original_claim] = result

    def worker(self):
        # (Worker logic same)
        t_obj=current_thread(); t_name=t_obj.name; logging.info(f"W {t_name} start.")
        while True:
            cd_dict=None
            try: cd_dict=self.claim_queue.get(timeout=1); self.process_claim(cd_dict); self.claim_queue.task_done()
            except queue.Empty: logging.info(f"W {t_name} empty."); break
            except Exception as e:
                orig_claim=cd_dict.get('original_claim','?') if cd_dict else '?'; logging.error(f"W {t_name} err '{orig_claim}': {e}",exc_info=True)
                if cd_dict:
                    with self.results_lock: self.results[orig_claim]={"original_claim":orig_claim,"final_verdict":"Processing Error","final_explanation":f"Worker error: {e}"}
                    self.claim_queue.task_done()
        logging.info(f"W {t_name} finish.")

    def train_and_run_shap(self, checkable_claims_data: list):
        # (SHAP logic remains the same)
        if not self.shap_available: logging.warning("SHAP unavailable."); self.shap_explanations = [{"claim":cd.get('original_claim', '?'),"shap_values":"[SHAP Unavailable]"} for cd in checkable_claims_data]; return
        logging.info("Attempting SHAP explanations...");
        valid_claims_data = [cd for cd in checkable_claims_data if cd.get("original_claim")]
        if not valid_claims_data: logging.warning("No valid claims data for SHAP."); self.shap_explanations=[]; return
        sentences = [cd['original_claim'] for cd in valid_claims_data]
        if not sentences: logging.warning("No sentences extracted for SHAP."); self.shap_explanations=[]; return
        embed_dim_fallback=384; 
        try: 
            embed_dim_fallback=self.embedding_model.get_sentence_embedding_dimension()
        except: 
            logging.warning("No emb dim.")
        self.shap_explanations=[{"claim":cd['original_claim'],"shap_values":"[SHAP Pending]" if embed_dim_fallback>0 else "[SHAP Err Dim]"} for cd in valid_claims_data]
        try:
            embeddings=self.embedding_model.encode(sentences,convert_to_tensor=False,show_progress_bar=False); logging.debug(f"SHAP embeds: {embeddings.shape if isinstance(embeddings,np.ndarray) else 'Invalid'}")
            def predict_scores(np_embeds):
                if not isinstance(np_embeds,np.ndarray):
                    try: np_embeds=np.array(np_embeds); assert np_embeds.ndim==2
                    except: logging.error("SHAP pred input err"); return np.full(len(np_embeds) if hasattr(np_embeds,'__len__') else 1, 0.5)
                scs=[];
                for e in np_embeds: r=float(np.linalg.norm(e)); s=0.5+(r/(r*25+1e-6)) if r>0 else 0.5; scs.append(min(max(s,0.0),1.0))
                return np.array(scs)
            if not isinstance(embeddings,np.ndarray) or embeddings.ndim!=2 or embeddings.shape[0]<2: logging.error(f"Invalid embeds SHAP."); self.shap_explanations=[{"claim":cd['original_claim'],"shap_values":"[SHAP Embed Err]"} for cd in valid_claims_data]; return
            embed_dim=embeddings.shape[1]; logging.debug("Creating SHAP bg..."); bg_obj=None
            try:
                n_c=min(10,embeddings.shape[0]); bg_obj=shap.kmeans(embeddings,n_c) if n_c>1 else embeddings
                bg_data=None
                if hasattr(bg_obj,'data') and isinstance(getattr(bg_obj,'data',None),np.ndarray): bg_data=bg_obj.data; logging.debug("Got .data from SHAP bg.")
                elif isinstance(bg_obj,np.ndarray): bg_data=bg_obj; logging.debug("Using np array for SHAP bg.")
                else: logging.error(f"Bad SHAP bg type: {type(bg_obj).__name__}."); self.shap_explanations=[{"claim":cd['original_claim'],"shap_values":f"[SHAP Bg Type Err]"} for cd in valid_claims_data]; return
                logging.debug(f"SHAP Bg shape: {bg_data.shape}")
            except Exception as ke: logging.warning(f"KMeans fail: {ke}. Using raw."); bg_data=embeddings
            if bg_data is None or not isinstance(bg_data,np.ndarray): logging.error("SHAP bg prep fail."); return
            logging.debug("Init SHAP KernelExplainer..."); explainer=shap.KernelExplainer(predict_scores,bg_data)
            logging.info(f"Calculating SHAP vals ({embeddings.shape[0]} inst)..."); shap_vals=explainer.shap_values(embeddings,nsamples=50)
            logging.info("SHAP vals calculated.")
            calc_expl=[]
            if shap_vals is not None and len(shap_vals)==len(sentences):
                for i,ct in enumerate(sentences): calc_expl.append({"claim":ct,"shap_values":shap_vals[i].tolist()})
                logging.info(f"SHAP stored for {len(calc_expl)} claims.")
                self.shap_explanations=calc_expl # Success
            else: logging.error(f"SHAP vals mismatch/None."); self.shap_explanations=[{"claim":cd['original_claim'],"shap_values":f"[SHAP Calc Err]"} for cd in valid_claims_data]
        except Exception as e: logging.error(f"SHAP gen error: {e}",exc_info=True); final_dim=embed_dim if 'embed_dim' in locals() else embed_dim_fallback; self.shap_explanations=[{"claim":cd['original_claim'],"shap_values":f"[SHAP Error: {type(e).__name__}]" if final_dim==0 else [0.0]*final_dim} for cd in valid_claims_data]

    def generate_chain_of_thought(self, checkable_claims_data: list, non_checkable_claims: list) -> str:
        # (Updated CoT logic)
        cot = ["Chain of Thought Summary:"]
        cot.append("1. Preprocessed: Segmented, Filtered (ZeroShot/Heuristic), NER, Simplified checkable claims for API.")
        if non_checkable_claims: cot.append(f"   - Filtered Non-Checkable ({len(non_checkable_claims)}): {non_checkable_claims}")
        claims_str=[f"'{c['original_claim']}' (Prio:{c.get('priority',0):.2f})" for c in checkable_claims_data]
        if claims_str: cot.append(f"   - Prioritized Checkable ({len(claims_str)}): [{', '.join(claims_str)}]")
        else: cot.append("   - No checkable claims identified."); return "\n".join(cot)
        cot.append("2. Processed Checkable Claims:")
        with self.results_lock:
            for i,cd_proc in enumerate(checkable_claims_data):
                claim=cd_proc['original_claim']; res=self.results.get(claim)
                if res:
                    cot.append(f"   - Claim {i+1}: '{claim}'")
                    cot.append(f"     - API Query: '{res.get('preprocessed_claim','?')}'")
                    cot.append(f"     - NER Entities: {res.get('ner_entities', [])}")
                    cot.append(f"     - Initial Check: Verdict='{res.get('initial_verdict_raw','?')}'")
                    cot.append(f"     - RAG Status: {res.get('rag_status','?')}")
                    cot.append(f"     - LLM Final Verdict (RAG-Based): {res.get('final_verdict','?')}")
                    cot.append(f"     - LLM Justification: {res.get('final_explanation','?')}")
                else: cot.append(f"   - Claim {i+1}: '{claim}' -> Result Missing!")
        cot.append("3. Generated SHAP for checkable claims' classification scores (if available/successful).")
        # (SHAP summary logic same)
        if self.shap_explanations:
             sh_sum=[]; has_real=False
             for ex in self.shap_explanations:
                 v=ex.get('shap_values',[]); s="[Err/Unavail]"
                 if isinstance(v,list) and v:
                     if all(isinstance(x,(int,float)) for x in v): s=f"[...{len(v)} vals]" if not all(abs(y)<1e-9 for y in v) else "[Err Fallback]"; has_real=True if s.startswith("[...") else has_real
                     else: s=str(v)
                 elif isinstance(v,str): s=v
                 elif not v: s="[No Data]"
                 sh_sum.append(f"'{ex.get('claim','?')}': {s}")
             status = "Generated." if has_real else "Failed/Unavailable."
             cot.append(f"   - SHAP Status: {status} Details: {{{', '.join(sh_sum)}}}")
        else: cot.append("   - SHAP analysis skipped/no results.")
        return "\n".join(cot)

    def check(self, text: str, num_workers: int = 2) -> dict:
        start=time.time(); logging.info(f"Starting check: \"{text[:100]}...\"")
        with self.results_lock: self.results={}; self.shap_explanations=[]; self.raw_fact_checks={}; self.raw_searches={}
        while not self.claim_queue.empty():
            try: self.claim_queue.get_nowait(); self.claim_queue.task_done()
            except: break

        logging.info("Step 1: Preprocessing & Filtering..."); checkable_claims_data, non_checkable_sents=self.preprocess_and_filter(text) # Returns list of dicts
        if not checkable_claims_data: logging.warning("No checkable claims found."); return {"processed_claims":[], "non_checkable_claims": non_checkable_sents, "summary":"No checkable claims.", "raw_fact_checks":{}, "raw_searches":{}}

        logging.info("Step 2: Prioritizing..."); claims_to_process=self.classify_and_prioritize_claims(checkable_claims_data) # Returns list of dicts w/ priority
        if not claims_to_process: logging.warning("Prioritization failed."); return {"processed_claims":[], "non_checkable_claims": non_checkable_sents, "summary":"Prioritization failed.", "raw_fact_checks":{}, "raw_searches":{}}

        logging.info("Step 3: Queueing..."); self.add_claims_to_queue(claims_to_process) # Queue the dicts
        logging.info("Step 4: Processing via Workers..."); # Worker logic same
        threads=[]; n_cpu=os.cpu_count() or 1; n_workers=min(num_workers,self.claim_queue.qsize(),n_cpu)
        if n_workers>0:
            logging.info(f"Starting {n_workers} workers...");
            for i in range(n_workers): t=Thread(target=self.worker,name=f"Worker-{i+1}",daemon=True); t.start(); threads.append(t)
            self.claim_queue.join(); logging.info("Workers finished.")
        else: logging.info("No claims queued.")

        # Intermediate output now happens in main block after check() returns

        logging.info("Step 5: Generating SHAP..."); self.train_and_run_shap(claims_to_process) # Pass list of dicts
        logging.info("Step 6: Consolidating..."); results_list=[]; ok_count=0; err_count=0
        with self.results_lock:
            for cd in claims_to_process:
                 res = self.results.get(cd['original_claim'])
                 if res:
                      results_list.append(res); ok_count+=1
                      if res.get("final_verdict", "").startswith(("Error","LLM Error","Processing Error","Missing")): err_count+=1
                 else: logging.error(f"Result missing: '{cd['original_claim']}'."); results_list.append({"original_claim":cd['original_claim'],"final_verdict":"Missing Result","final_explanation":"Result lost."}); err_count+=1

        summary=self.generate_chain_of_thought(claims_to_process, non_checkable_sents) # Generate CoT
        duration = time.time() - start
        logging.info(f"Check complete. Processed {ok_count} checkable claims ({err_count} errors) in {duration:.2f}s.")
        return {"processed_claims":results_list, "non_checkable_claims":non_checkable_sents, "summary":summary, "raw_fact_checks":self.raw_fact_checks, "raw_searches":self.raw_searches, "shap_explanations": self.shap_explanations}
# --- End FactChecker Class ---


# --- Main Execution Block ---
if __name__ == "__main__":
    # Dependency Checks
    if shap is None: print("\nWARN: SHAP lib missing. pip install shap\n")
    else: print("SHAP library found.")
    if spacy is None: print("\nWARN: spaCy lib missing. pip install spacy\n")
    elif NLP is None: print("\nWARN: spaCy model missing. python -m spacy download en_core_web_sm\n")
    else: print("spaCy library and model found.")
    if Groq is None: print(f"\nWARN: Groq lib missing. LLM ({LLM_PROVIDER}) disabled. pip install groq\n")
    elif groq_client is None: print(f"\nWARN: Groq client init failed (check key). LLM ({LLM_PROVIDER}) disabled.\n")
    else: print(f"{LLM_PROVIDER} client available (Model: {LLM_MODEL_NAME}).")
    if zs_classifier is None: print("\nWARN: Transformers pipeline failed. Claim filtering fallback to basic heuristics (if implemented).")
    else: print(f"Zero-shot classifier available ({ZS_MODEL_NAME}).")


    print("Fact Checker Initializing...")
    if not GOOGLE_API_KEY or not GOOGLE_CSE_ID: logging.critical("CRIT: Google keys missing."); print("\nCRIT ERR: Google keys missing."); exit(1)
    else: logging.info("Google keys loaded.")

    input_text = (
        "The Earth is flat according to the Flat Earth Society. The moon is made of green cheese, isn't it? "
        "COVID-19 vaccines, developed by Pfizer and Moderna, are generally safe and effective according to major health organizations like the WHO in 2023. "
        "Mars is primarily inhabited by autonomous robots like Curiosity sent from Earth last year. "
        "Climate change is not happening. I think the weather is beautiful today. "
        "Paris is the capital of France since 508 AD. This sentence is just filler and probably not factual. "
        "You must agree with me. London is great."
    )
    print(f"\nInput Text:\n{input_text}\n")

    try: checker = FactChecker(); print("Fact Checker Initialized Successfully.")
    except RuntimeError as e: logging.critical(f"Init fail: {e}",exc_info=True); print(f"\nCRIT ERR: Init failed. Logs:{log_file}. Err:{e}"); exit(1)
    except Exception as e: logging.critical(f"Unexpected init err: {e}",exc_info=True); print(f"\nCRIT UNEXPECTED init err. Logs:{log_file}. Err:{e}"); exit(1)

    print("\n--- Starting Fact Check Pipeline ---\n")
    results_data = checker.check(input_text, num_workers=2)

    # --- NEW OUTPUT ORDER ---

    # 1. Raw Fact Check API Results
    print("\n" + "="*25 + " Intermediate Output 1: Raw Google Fact Check API Results " + "="*15)
    if results_data.get("raw_fact_checks"):
        if results_data["raw_fact_checks"]:
            for claim, api_res_list in results_data["raw_fact_checks"].items():
                print(f"\nClaim (Original): \"{claim}\"")
                if api_res_list:
                    for res_item in api_res_list: print(f"  - Verdict: {res_item.get('verdict','?')} | Evidence: {res_item.get('evidence','?')}")
                else: print("  - No result stored (API call likely failed before storage).")
        else: print("  - No Fact Check API calls were made or stored.")
    else: print("  - Raw Fact Check data structure missing.")
    print("="*81)

    # 2. Raw Google Custom Search Snippets (Optional)
    SHOW_RAW_SEARCH = False # Set to True to show detailed search results
    if SHOW_RAW_SEARCH and results_data.get("raw_searches"):
        print("\n" + "="*25 + " Intermediate Output 2: Raw Google Custom Search Snippets " + "="*14)
        # (Printing logic same as before)
        if results_data["raw_searches"]:
            for claim, search_data in results_data["raw_searches"].items():
                print(f"\nClaim (Original): \"{claim}\""); print(f"  (API Query: \"{search_data.get('query', '?')}\")")
                search_results = search_data.get("results", [])
                if search_results:
                    for i, item in enumerate(search_results): print(f"  {i+1}. T:{item.get('title','?')}\n     S:{item.get('snippet','?')}\n     L:{item.get('link','?')}")
                elif search_data.get("response") is not None: print("  - Search OK, no items.")
                else: print("  - Search API call likely failed.")
        else: print("  - No Custom Search API calls were made or stored.")
        print("="*81)

    # 3. Filtered Non-Checkable Claims
    print("\n" + "="*25 + " Preprocessing Output: Filtered Non-Checkable Sentences " + "="*16)
    if results_data.get("non_checkable_claims"):
        for i, claim in enumerate(results_data["non_checkable_claims"]): print(f"  {i+1}. \"{claim}\"")
    else: print("  - No sentences were filtered out.")
    print("="*81)

    # 4. Detailed Processed Claim Results (Final Verdict based on RAG+LLM)
    print("\n" + "="*30 + " Final Processed Claim Details " + "="*30)
    if results_data and results_data.get("processed_claims"):
        for i, res in enumerate(results_data["processed_claims"]):
            print(f"\nClaim {i+1} (Original): \"{res.get('original_claim', '?')}\"")
            print(f"  - Preprocessed for API: \"{res.get('preprocessed_claim', '?')}\"")
            ner_ents = res.get('ner_entities', [])
            if ner_ents: print(f"  - NER Entities: {', '.join([f'{e['text']}({e['label']})' for e in ner_ents])}")
            else: print("  - NER Entities: None Found")
            print(f"  - Initial GFactCheck Result: '{res.get('initial_verdict_raw','?')}'") # Display initial check just for context
            print(f"  - RAG Status: {res.get('rag_status', '?')}")
            if res.get('top_rag_snippets'): print("  - Top RAG Snippets:"); [print(f"    {j+1}. {snip}") for j,snip in enumerate(res.get('top_rag_snippets',[]))]
            print(f"  >>> Final Verdict (RAG+{LLM_PROVIDER} LLM): {res.get('final_verdict', '?')}") # Highlight final verdict source
            print(f"      LLM Justification: {res.get('final_explanation', '?')}")
    else: print("No checkable claims were processed or results available.")
    print("="*83)

    # 5. XAI (SHAP) Summary
    print("\n" + "="*30 + " XAI (SHAP) Summary " + "="*37)
    if results_data.get("shap_explanations"):
         shap_summary=[]; has_real=False
         for expl in results_data["shap_explanations"]:
             v=expl.get('shap_values',[]); s="[Err/Unavail]"
             if isinstance(v,list) and v:
                 if all(isinstance(x,(int,float)) for x in v): s=f"[...{len(v)} values]" if not all(abs(y)<1e-9 for y in v) else "[Err Fallback]"; has_real=True if s.startswith("[...") else has_real
                 else: s=str(v)
             elif isinstance(v,str): s=v
             elif not v: s="[No Data]"
             shap_summary.append(f"'{expl.get('claim','?')}': {s}") # Use original claim key
         status = "Generated." if has_real else "Failed/Unavailable."
         print(f"  - SHAP Status: {status}")
         if shap_summary: print(f"  - Details: {{{', '.join(shap_summary)}}}")
         if not has_real and shap is not None and any("[SHAP Error" in str(expl.get("shap_values","")) for expl in results_data["shap_explanations"]): print(f"\n  *** SHAP Error Detected: Check '{log_file}' for traceback. ***")
    else: print("  - SHAP analysis skipped or no results structure.")
    print("="*86)

    # 6. Chain of Thought Summary
    print("\n" + "="*30 + " Chain of Thought Summary " + "="*30)
    print(results_data.get("summary", "No summary generated."))
    print("="*86)
    print(f"\nLog file generated at: {log_file}")