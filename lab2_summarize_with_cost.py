import os
import requests
import docx
from openai import OpenAI
from dotenv import load_dotenv
import io

# --- Configuration ---
load_dotenv()
client = OpenAI()
DOCX_URL = "https://static.alta3.com/files/coke_risk_factors.docx"
PDF_URL = "https://static.alta3.com/files/coke_risk.pdf"
MODEL_TO_USE = "gpt-4-turbo-preview"

# --- Cost Calculation Section ---
# Prices per 1,000,000 tokens for gpt-4-turbo-preview
COST_PER_INPUT_TOKEN = 10.00 / 1_000_000
COST_PER_OUTPUT_TOKEN = 30.00 / 1_000_000

def calculate_cost(usage):
    """Calculates the cost of an API call based on token usage."""
    input_cost = usage.prompt_tokens * COST_PER_INPUT_TOKEN
    output_cost = usage.completion_tokens * COST_PER_OUTPUT_TOKEN
    total_cost = input_cost + output_cost
    return total_cost

def download_and_extract_text(url):
    """Downloads a .docx file and extracts its text content."""
    print(f"--> Downloading document from {url}...")
    try:
        response = requests.get(url)
        response.raise_for_status()
        doc_stream = io.BytesIO(response.content)
        print("--> Extracting text from document...")
        doc = docx.Document(doc_stream)
        full_text = [para.text for para in doc.paragraphs]
        return '\n'.join(full_text)
    except Exception as e:
        print(f"    ERROR: {e}")
        return None

def summarize_text(text_content):
    """Sends the text to the OpenAI API and returns the full response object."""
    print("--> Sending text to OpenAI for summarization...")
    prompt = f"""
    Act as a senior financial analyst. The following text contains the "Risk Factors" section from The Coca-Cola Company's 2024 10-K filing.
    Your task is to produce a three-part summary for an investment committee:
    1.  **Top 3 Thematic Risks:** Identify and explain the three most significant, overarching risk themes.
    2.  **Emerging Risk:** Identify one risk that appears to be emerging or increasing in importance.
    3.  **Overall Tone:** Describe the overall tone of the risk disclosure.

    You can verify your summary against the original source document here: {PDF_URL}

    Here is the text: --- {text_content} ---
    """
    try:
        # We use a non-streaming call here to get the token usage data
        response = client.chat.completions.create(
            model=MODEL_TO_USE,
            messages=[{"role": "user", "content": prompt}]
        )
        return response
    except Exception as e:
        print(f"    ERROR: OpenAI API call failed. {e}")
        return None

if __name__ == "__main__":
    print("#"*70)
    print("##" + " Lab 2: Real-World 10-K Summarization & Cost Analysis".center(66) + "##")
    print("#"*70)
    
    risk_factors_text = download_and_extract_text(DOCX_URL)
    
    if risk_factors_text:
        api_response = summarize_text(risk_factors_text)
        if api_response:
            summary = api_response.choices[0].message.content
            usage = api_response.usage
            
            print("\n" + "="*70)
            print("||" + " AI-Generated Executive Summary".center(66) + "||")
            print("="*70)
            print(summary)
            
            print("\n" + "="*70)
            print("||" + " Cost Analysis".center(66) + "||")
            print("="*70)
            print(f"  - Model Used: {MODEL_TO_USE}")
            print(f"  - Input Tokens: {usage.prompt_tokens}")
            print(f"  - Output Tokens: {usage.completion_tokens}")
            print(f"  - Total Tokens: {usage.total_tokens}")
            
            cost = calculate_cost(usage)
            print(f"  - Cost for this run: ${cost:.6f}")
        
            print("="*70)

