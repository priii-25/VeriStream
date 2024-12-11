import requests
from dotenv import load_dotenv
import os

load_dotenv()
API_KEY = os.getenv("API_KEY")
BASE_URL = "https://factchecktools.googleapis.com/v1alpha1/claims:search"

def search_fact_check(claim: str):
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

if __name__ == "__main__":
    claim_text = "COVID-19 vaccines are ineffective."
    result = search_fact_check(claim_text)

    if "error" in result:
        print(f"Error: {result['error']}")
    else:
        print("Fact-Checked Results:")
        for claim in result.get("claims", []):
            print(f"Claim: {claim.get('text')}")
            print(f"Claimant: {claim.get('claimant')}")
            print(f"Claimed Date: {claim.get('claimDate')}")
            print("Review:")
            for review in claim.get("claimReview", []):
                print(f"  Publisher: {review.get('publisher', {}).get('name')}")
                print(f"  Rating: {review.get('textualRating')}")
                print(f"  URL: {review.get('url')}")
            print("-" * 50)