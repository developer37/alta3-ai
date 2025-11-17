import os
import json
import requests
from openai import OpenAI
from dotenv import load_dotenv

# --- Setup ---
load_dotenv()
openai_client = OpenAI()
# Updated to use a model specifically fine-tuned on financial news
HF_API_URL = "https://router.huggingface.co/hf-inference/models/ProsusAI/finbert"
HF_API_KEY = os.environ.get("HUGGING_FACE_API_KEY")

def print_header(title):
    print("\n" + "#"*60)
    print(f"## {title.center(56)} ##")
    print("#"*60)

def analyze_sentiment(text_content):
    print_header("Part A: Comparative Sentiment Analysis")
    
    # OpenAI Analysis
    try:
        prompt = f'Analyze the sentiment of the following news article. Respond with a single word: "Positive", "Negative", or "Neutral".\n\n{text_content}'
        response = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}]
        )
        print(">>> OpenAI (GPT-3.5) Sentiment:", response.choices[0].message.content)
    except Exception as e:
        print(f">>> OpenAI Error: {e}")

    # Hugging Face Analysis with Enhanced Error Handling
    if not HF_API_KEY:
        print(">>> Hugging Face: API key not set in .env file. Skipping.")
        return
    
    print("\n>>> Calling Hugging Face (may take a moment to warm up)...")
    try:
        headers = {"Authorization": f"Bearer {HF_API_KEY}"}
        payload = {"inputs": text_content}
        response = requests.post(HF_API_URL, headers=headers, json=payload)
        
        # Raise an exception for bad status codes (4xx or 5xx)
        response.raise_for_status()
        
        # If we get here, the status code was 200
        result = response.json()
        print(">>> Hugging Face (FinBERT) Sentiment:", result)

    except requests.exceptions.HTTPError as http_err:
        print(f">>> Hugging Face HTTP Error: {http_err}")
        print(f"    Raw Response Text: {response.text}")
    except requests.exceptions.JSONDecodeError:
         print(">>> Hugging Face API Error: The API returned a non-JSON response.")
         print(f"    Raw Response Text: {response.text}")
    except Exception as e:
        print(f">>> An unexpected error occurred with Hugging Face: {e}")

def extract_structured_data(text_content):
    print_header("Part B: Structured Data Extraction (JSON)")
    
    prompt = f"""
    Extract the key financial and strategic data points from the following news article.
    Your response MUST be a valid JSON object with the following keys: "company_name", "ticker", "revenue_reported", "key_positive_driver", "key_negative_concerns".

    Article:
    ---
    {text_content}
    ---
    """
    try:
        response = openai_client.chat.completions.create(
            model="gpt-4-turbo-preview",
            response_format={"type": "json_object"},
            messages=[{"role": "user", "content": prompt}]
        )
        parsed_json = json.loads(response.choices[0].message.content)
        print(json.dumps(parsed_json, indent=4))
    except Exception as e:
        print(f"Error extracting JSON: {e}")

if __name__ == "__main__":
    try:
        with open('news_article.txt', 'r') as file:
            article_text = file.read()
        analyze_sentiment(article_text)
        extract_structured_data(article_text)
    except FileNotFoundError:
        print("Error: news_article.txt not found.")

