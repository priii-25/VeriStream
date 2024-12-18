import os
import faiss
import numpy as np
from transformers import AutoTokenizer, AutoModel, AutoModelForSeq2SeqLM
from typing import List, Dict, Any
import requests

API_KEY = os.getenv("API_KEY")
FACT_CHECK_URL = "https://factchecktools.googleapis.com/v1alpha1/claims:search"

def search_fact_check(claim: str) -> List[str]:
    """
    Fetch fact-checking results for a given claim using the Google Fact Check API.
    Args:
        claim (str): Claim to fact-check.
    Returns:
        List[str]: List of fact-checked statements and their reviews.
    """
    params = {
        "query": claim,
        "key": API_KEY
    }
    try:
        response = requests.get(FACT_CHECK_URL, params=params)
        response.raise_for_status()
        data = response.json()
        documents = []
        for claim_item in data.get("claims", []):
            text = claim_item.get("text", "")
            for review in claim_item.get("claimReview", []):
                publisher = review.get("publisher", {}).get("name", "Unknown Publisher")
                rating = review.get("textualRating", "No Rating")
                url = review.get("url", "No URL")
                documents.append(f"{text} - Reviewed by {publisher} ({rating}): {url}")
        return documents
    except requests.exceptions.RequestException as e:
        print(f"Error during fact-checking: {e}")
        return []

class RAGPipeline:
    def __init__(self, model_name: str, retriever_model_name: str, documents: List[str]):
        """
        Initialize RAG Pipeline with document embedding and retrieval.
        Args:
            model_name (str): Hugging Face model for LLM.
            retriever_model_name (str): Hugging Face model for dense retrieval.
            documents (List[str]): Corpus of documents for knowledge retrieval.
        """
        self.documents = documents
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.llm = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        
        self.retriever_tokenizer = AutoTokenizer.from_pretrained(retriever_model_name)
        self.retriever_model = AutoModel.from_pretrained(retriever_model_name)
        
        self.index, self.embedded_docs = self.build_index(documents)

    def embed_text(self, text: str) -> np.ndarray:
        """
        Embed text using the retriever model.
        Args:
            text (str): Input text to embed.
        Returns:
            np.ndarray: Embedded representation of the text.
        """
        inputs = self.retriever_tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = self.retriever_model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

    def build_index(self, documents: List[str]) -> (faiss.IndexFlatL2, List[str]):
        """
        Build FAISS index for document embeddings.
        Args:
            documents (List[str]): Corpus of documents.
        Returns:
            faiss.IndexFlatL2: FAISS index for retrieval.
            List[str]: Original document list (to fetch results).
        """
        embeddings = [self.embed_text(doc) for doc in documents]
        embedding_matrix = np.stack(embeddings)
        index = faiss.IndexFlatL2(embedding_matrix.shape[1])
        index.add(embedding_matrix)
        return index, documents

    def retrieve_documents(self, query: str, top_k: int = 3) -> List[str]:
        """
        Retrieve top-k relevant documents for a given query.
        Args:
            query (str): Query string.
            top_k (int): Number of top documents to retrieve.
        Returns:
            List[str]: Top-k relevant documents.
        """
        query_embedding = self.embed_text(query).reshape(1, -1)
        distances, indices = self.index.search(query_embedding, top_k)
        return [self.documents[i] for i in indices[0]]

    def generate_response(self, query: str, context: List[str]) -> str:
        """
        Generate response using LLM with the retrieved context.
        Args:
            query (str): Query string.
            context (List[str]): Retrieved contextual documents.
        Returns:
            str: Generated response.
        """
        context_text = " ".join(context)
        input_text = f"Context: {context_text}\n\nQuery: {query}\n\nAnswer:"
        inputs = self.tokenizer(input_text, return_tensors="pt", truncation=True, padding=True)
        outputs = self.llm.generate(**inputs, max_length=256, num_return_sequences=1)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

def main():
    claims = [
        "Narendra Modi is the best prime minister of India",
    ]

    all_documents = []
    for claim in claims:
        documents = search_fact_check(claim)
        all_documents.extend(documents)

    if not all_documents:
        print("No fact-checked documents found. Please refine the queries.")
        return

    model_name = "google/flan-t5-base" 
    retriever_model_name = "sentence-transformers/all-mpnet-base-v2"
    rag_pipeline = RAGPipeline(model_name=model_name, retriever_model_name=retriever_model_name, documents=all_documents)
    
    for claim in claims:
        print(f"\n{'='*50}\nQuery: {claim}\n{'='*50}")
        result = rag_pipeline.retrieve_documents(claim)
        response = rag_pipeline.generate_response(claim, result)
        print(f"Retrieved Documents: {result}")
        print(f"Generated Response: {response}")

if __name__ == "__main__":
    main()
