import requests
import re
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

sentiment_model_name = "cardiffnlp/twitter-roberta-base-sentiment"
sentiment_pipeline = pipeline("sentiment-analysis", model=sentiment_model_name)

FACT_CHECK_API = "https://factchecktools.googleapis.com/v1alpha1/claims:search"
API_KEY = ""

def perform_fact_check(query):
    """Queries the Google Fact-Check API."""
    response = requests.get(
        FACT_CHECK_API,
        params={"query": query, "key": API_KEY},
    )
    if response.status_code == 200:
        data = response.json()
        if "claims" in data:
            return data["claims"]
        else:
            return []
    else:
        print("Error contacting fact-check API", response.status_code)
        return []

def extract_factual_statements(text):
    """Simplistic factual claim extraction using regex."""
    return re.findall(r"[^.!?]+[.!?]", text)

def compute_manipulation_score(sentiment, fact_check_results):
    """Compute a score based on sentiment and fact-check results."""
    sentiment_score = 1 if sentiment["label"] == "LABEL_0" else (0.5 if sentiment["label"] == "LABEL_1" else 0)
    fact_score = 1 if not fact_check_results else 0.5

    return round((sentiment_score + fact_score) / 2, 2)

def analyze_text(text):
    """Performs sentiment analysis and fact-checking on input text."""
    print("Analyzing Text...")

    sentiment_results = sentiment_pipeline(text)
    sentiment = sentiment_results[0]

    factual_statements = extract_factual_statements(text)
    fact_check_results = []
    for statement in factual_statements:
        fact_check_results.extend(perform_fact_check(statement))

    risk_score = compute_manipulation_score(sentiment, fact_check_results)

    return {
        "text": text,
        "sentiment": sentiment,
        "fact_check_results": fact_check_results,
        "manipulation_risk_score": risk_score,
    }

if __name__ == "__main__":
    input_text = "Breaking: Deadly virus spreading rapidly in your city!"
    result = analyze_text(input_text)

    print("\nAnalysis Results:")
    print("Sentiment:", result["sentiment"])
    print("Fact-Check Results:")
    for claim in result["fact_check_results"]:
        print(" -", claim["text"], "(Rating:", claim["claimReview"][0]["textualRating"], ")")
    print("Manipulation Risk Score:", result["manipulation_risk_score"])