import requests
import re
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

sentiment_model_name = "cardiffnlp/twitter-roberta-base-sentiment"
sentiment_pipeline = pipeline("sentiment-analysis", model=sentiment_model_name)

FACT_CHECK_API = "https://factchecktools.googleapis.com/v1alpha1/claims:search"
API_KEY = ""
EMOTIONAL_TRIGGERS = [
    r"breaking news",
    r"fear of",
    r"unprecedented",
    r"you won’t believe",
    r"urgent",
    r"deadly",
    r"shocking",
    r"limited time offer",
    r"once in a lifetime opportunity",
    r"do not miss out",
    r"last chance",
    r"exclusive",
    r"highly anticipated",
    r"must see",
    r"never before seen",
    r"immediately",
    r"critical",
    r"don’t wait",
    r"you’re running out of time",
    r"emergency",
    r"breaking updates",
    r"unbelievable",
    r"catastrophic",
    r"life-changing",
    r"the clock is ticking",
    r"extremely rare",
    r"could be your last chance",
    r"your survival depends on this",
    r"could be too late",
    r"heart-stopping",
    r"surprising",
    r"stay alert",
    r"dangerous",
    r"groundbreaking",
    r"never seen before",
    r"deadly impact",
    r"warning",
    r"crisis",
    r"unexpected",
    r"urgent response needed",
    r"life-threatening",
    r"something is wrong",
    r"explosive",
    r"shocking twist",
    r"urgent action required",
    r"trending",
    r"disaster",
    r"you won’t believe what happens next",
    r"last-minute decision",
    r"urgent situation",
    r"breaking headlines",
    r"unexpected turn of events",
    r"critical alert",
    r"groundbreaking discovery",
    r"must-act-now",
    r"biggest threat",
    r"unpredictable",
    r"stay tuned",
    r"new revelation",
]

STEREOTYPE_PATTERNS = [
    r"all \w+s are",
    r"\w+ people always",
    r"typical \w+ behavior",
    r"women can’t",
    r"men are always",
    r"every \w+ thinks",
    r"all \w+ men are",
    r"\w+ women can’t",
    r"you know how \w+s are",
    r"only \w+ people do that",
    r"all \w+s like to",
    r"every \w+ is",
    r"\w+ people never",
    r"\w+ men always",
    r"all \w+ women are",
    r"typical \w+ woman",
    r"they’re all \w+s",
    r"every \w+ thinks this way",
    r"the typical \w+ trait",
    r"\w+s never \w+",
    r"only \w+ men can",
    r"all \w+s hate",
    r"most \w+s believe",
    r"all \w+ boys are",
    r"\w+s can’t \w+",
    r"you won’t find a \w+ who",
    r"\w+s are always \w+",
    r"\w+ women never",
    r"all \w+ men think",
    r"it’s in a \w+ nature to",
    r"every \w+ woman must",
    r"all \w+ men are \w+",
    r"this is what \w+s do",
    r"only a \w+ would do that",
    r"all \w+s are always \w+",
    r"\w+s are better than \w+s",
    r"everyone knows that \w+s",
    r"every \w+ wants to",
    r"they say all \w+s are",
    r"you can’t trust a \w+ to",
    r"all \w+ girls are",
    r"all \w+s never care",
    r"it’s typical for \w+s to",
    r"there’s nothing like a \w+",
    r"all \w+s love to \w+",
    r"men always \w+ better than women",
    r"women always \w+",
    r"it’s common for \w+s to",
    r"you won’t find a \w+ who doesn’t",
    r"\w+s don’t care about",
    r"men don’t \w+ like women do",
    r"all \w+ girls want to",
    r"\w+s think only about \w+",
    r"all \w+s believe",
    r"\w+s never listen",
    r"you can always count on \w+s to",
    r"all \w+s think the same way",
]


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

def detect_emotional_triggers(text):
    """Detect emotionally manipulative phrases."""
    return [trigger for trigger in EMOTIONAL_TRIGGERS if re.search(trigger, text, re.IGNORECASE)]

def detect_stereotypes(text):
    """Detect stereotypes in the text."""
    return [pattern for pattern in STEREOTYPE_PATTERNS if re.search(pattern, text, re.IGNORECASE)]

def compute_manipulation_score(sentiment, fact_check_results, emotional_triggers, stereotypes):
    """Compute a score based on multiple factors."""
    sentiment_score = 1 if sentiment["label"] == "LABEL_0" else (0.5 if sentiment["label"] == "LABEL_1" else 0)
    fact_score = 1 if not fact_check_results else 0.5
    trigger_score = 1 if emotional_triggers else 0.5
    stereotype_score = 1 if stereotypes else 0.5

    return round((sentiment_score + fact_score + trigger_score + stereotype_score) / 4, 2)

def analyze_text(text):
    """Performs sentiment analysis, fact-checking, and manipulation detection on input text."""
    print("Analyzing Text...")

    sentiment_results = sentiment_pipeline(text)
    sentiment = sentiment_results[0]

    factual_statements = extract_factual_statements(text)
    fact_check_results = []
    for statement in factual_statements:
        fact_check_results.extend(perform_fact_check(statement))

    emotional_triggers = detect_emotional_triggers(text)

    stereotypes = detect_stereotypes(text)

    risk_score = compute_manipulation_score(sentiment, fact_check_results, emotional_triggers, stereotypes)

    return {
        "text": text,
        "sentiment": sentiment,
        "fact_check_results": fact_check_results,
        "emotional_triggers": emotional_triggers,
        "stereotypes": stereotypes,
        "manipulation_risk_score": risk_score,
    }

if __name__ == "__main__":
    input_text = "breaking news: women always lead."
    result = analyze_text(input_text)

    print("\nAnalysis Results:")
    print("Sentiment:", result["sentiment"])
    print("Fact-Check Results:")
    for claim in result["fact_check_results"]:
        print(" -", claim["text"], "(Rating:", claim["claimReview"][0]["textualRating"], ")")
    print("Emotional Triggers:", result["emotional_triggers"])
    print("Stereotypes Detected:", result["stereotypes"])
    print("Manipulation Risk Score:", result["manipulation_risk_score"])