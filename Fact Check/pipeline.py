import os
import requests

# For demonstration: Hugging Face for NER and classification
from transformers import pipeline

# LangChain imports for RAG and/or LLM usage
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFacePipeline
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.docstore.document import Document
from transformers import pipeline

API_KEY ="AIzaSyA_o1LhjLi2vIi1sgTUStzeTBiUFrGOLYI"
BASE_URL = "https://factchecktools.googleapis.com/v1alpha1/claims:search"

def search_fact_check(claim: str):
    """
    Queries the Google FactCheck Tools API to find fact-checking information
    related to a specific claim.
    """
    params = {
        "query": claim,
        "key": API_KEY
    }
    try:
        response = requests.get(BASE_URL, params=params)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        return {"error": str(e)}

def main_claim_pipeline(claim_text: str):
    """
    Main function:
      1. Retrieves fact-check data with the FactCheck Tools API.
      2. Performs NER on the claim to identify entities.
      3. Uses zero-shot classification to produce a "true" / "false" result with a confidence score.
      4. Integrates RAG + an LLM via LangChain to demonstrate combined outputs.
    """

    # --- Step 1: Retrieve fact-check data ---
    factcheck_result = search_fact_check(claim_text)
    if "error" in factcheck_result:
        print(f"Error accessing FactCheck Tools API: {factcheck_result['error']}")
        return

    print("Fact-Checked Results from Google FactCheck Tools:")
    claims_data = factcheck_result.get("claims", [])
    if not claims_data:
        print("No claims found for the given query.\n")
    else:
        for claim in claims_data:
            text = claim.get('text', 'N/A')
            claimant = claim.get('claimant', 'N/A')
            claim_date = claim.get('claimDate', 'N/A')
            print(f"Claim: {text}")
            print(f"Claimant: {claimant}")
            print(f"Claimed Date: {claim_date}")
            for review in claim.get('claimReview', []):
                publisher_name = review.get('publisher', {}).get('name', 'N/A')
                textual_rating = review.get('textualRating', 'N/A')
                review_url = review.get('url', 'N/A')
                print(f"  Publisher: {publisher_name}")
                print(f"  Rating: {textual_rating}")
                print(f"  URL: {review_url}")
            print("-" * 50)

    # --- Step 2: Perform NER on the user-submitted claim ---
    ner_pipeline = pipeline(
        task="ner",
        model="dslim/bert-base-NER",
        aggregation_strategy="simple"
    )
    ner_results = ner_pipeline(claim_text)
    print("\nNamed Entities in the Claim Text:")
    for entity in ner_results:
        entity_text = entity['word']
        entity_type = entity['entity_group']
        print(f"  - {entity_text} ({entity_type})")

    # --- Step 3: Classify the claim as 'true' or 'false' ---
    zero_shot_classifier = pipeline(
        task="zero-shot-classification",
        model="facebook/bart-large-mnli"
    )
    possible_labels = ["true claim", "false claim"]
    classification_result = zero_shot_classifier(claim_text, possible_labels)

    top_label = classification_result["labels"][0]
    top_score = classification_result["scores"][0]

    print("\nClaim Classification:")
    if top_label == "true claim":
        print(f"Inferred to be TRUE with confidence {top_score:.2f}")
    else:
        print(f"Inferred to be FALSE with confidence {top_score:.2f}")

    # --- Step 4: Demonstrate using LangChain for RAG or LLM output ---
    if claims_data:
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        docs = []
        for claim_item in claims_data:
            claim_text_data = claim_item.get("text", "")
            docs.append(Document(page_content=claim_text_data))

        vectorstore = FAISS.from_documents(docs, embeddings)

        access_token = "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"
        llama_pipeline = pipeline("text-generation",model="meta-llama/Llama-2-7b",use_auth_token=access_token)

        qa_chain = RetrievalQA.from_chain_type(
            llm=llama_pipeline,
            chain_type="stuff",
            retriever=vectorstore.as_retriever()
        )

        question = f"What do we know about the claim: '{claim_text}' based on the retrieved fact-check data?"
        final_answer = qa_chain.run(question)
        print("\nRAG-based answer from LLM:\n", final_answer)

if __name__ == "__main__":
    user_claim = "COVID-19 vaccines are ineffective."
    main_claim_pipeline(user_claim)