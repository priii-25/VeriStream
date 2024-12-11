import requests
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, pipeline
from haystack.document_stores import FAISSDocumentStore
import faiss
from haystack.nodes import DensePassageRetriever, FARMReader
from haystack.pipelines import ExtractiveQAPipeline
from dotenv import load_dotenv
import os
load_dotenv()
class FactChecker:
    def __init__(self, api_key, fact_check_api_url):
        self.api_key = api_key
        self.fact_check_api_url = fact_check_api_url
        self.tokenizer = AutoTokenizer.from_pretrained("deepset/bert-base-cased-squad2")
        self.model = AutoModelForQuestionAnswering.from_pretrained("deepset/bert-base-cased-squad2")
        self.qa_pipeline = pipeline("question-answering", model=self.model, tokenizer=self.tokenizer)
        
        self.document_store = FAISSDocumentStore(faiss_index_factory_str="Flat")
        self.retriever = DensePassageRetriever(
            document_store=self.document_store,
            query_embedding_model="facebook/dpr-question_encoder-single-nq-base",
            passage_embedding_model="facebook/dpr-ctx_encoder-single-nq-base",
        )
        self.reader = FARMReader(model_name_or_path="deepset/bert-base-cased-squad2", use_gpu=False)
        self.pipeline = ExtractiveQAPipeline(reader=self.reader, retriever=self.retriever)

    def fetch_fact_check_results(self, claim):
        """
        Fetch results from external Fact-Check API.
        """
        params = {"query": claim, "key": self.api_key}
        response = requests.get(self.fact_check_api_url, params=params)
        if response.status_code == 200:
            data = response.json()
            results = []
            for claim in data.get("claims", []):
                results.append({
                    "content": claim.get("text", ""),
                    "meta": {"source": "fact_check_api", "rating": claim.get("claimReview", [{}])[0].get("textualRating", "unknown")},
                })
            return results
        else:
            raise ValueError(f"API Error: {response.status_code} - {response.text}")

    def add_to_document_store(self, docs):
        """
        Add documents to FAISS document store.
        """
        self.document_store.write_documents(docs)
        self.document_store.update_embeddings(self.retriever)

    def validate_claim(self, claim):
        """
        Validate the claim using RAG and pre-trained BERT models.
        """
        retrieved_docs = self.pipeline.run(query=claim, params={"Retriever": {"top_k": 3}})
        context = " ".join([doc["content"] for doc in retrieved_docs["documents"]])

        validation_result = self.qa_pipeline({"question": claim, "context": context})
        return {
            "claim": claim,
            "score": validation_result["score"],
            "answer": validation_result["answer"],
            "context": context,
        }

    def run_fact_check(self, claim):
        """
        End-to-end Fact-Check.
        """
        api_results = self.fetch_fact_check_results(claim)
        print(f"API Results Fetched: {api_results}")

        self.add_to_document_store(api_results)

        validation_result = self.validate_claim(claim)

        return {
            "claim": claim,
            "fact_check_api_results": api_results,
            "validation_result": validation_result,
        }


if __name__ == "__main__":
    API_KEY = os.getenv("API_KEY")
    fact_check_api_url = "https://factchecktools.googleapis.com/v1alpha1/claims:search"
    fact_checker = FactChecker(API_KEY, fact_check_api_url)

    claim = "COVID-19 vaccines increase the risk of infection."
    results = fact_checker.run_fact_check(claim)

    print("Fact-Check Results:")
    print(results)