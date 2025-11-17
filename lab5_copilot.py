import os
import json
from openai import OpenAI
from dotenv import load_dotenv

# --- Setup ---
load_dotenv()
client = OpenAI()

def load_client_data(filepath):
    """Loads client data from a JSON file."""
    try:
        with open(filepath, 'r') as file:
            return json.load(file)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error loading client data: {e}")
        return None

def generate_advisor_draft(client_data):
    """Generates and streams a draft email for a financial advisor."""
    
    portfolio_summary = ", ".join(f"{ticker} ({details['sector']})" for ticker, details in client_data.get('portfolio', {}).items())

    prompt = f"""
    You are an AI assistant for a licensed financial advisor. Your task is to draft an email response to a client based on their profile and latest inquiry.

    **CRITICAL RULES FOR COMPLIANCE:**
    1.  **DO NOT** provide any investment advice.
    2.  **DO NOT** suggest buying or selling any specific securities. Do not even hint at it.
    3.  **DO NOT** make any performance guarantees or predictions.
    4.  Your primary goal is to acknowledge the client's concern and proactively schedule a meeting with their human advisor.
    5.  You MUST include a standard compliance disclaimer at the end.

    **CLIENT PROFILE:**
    - Name: {client_data.get('client_name')}
    - Risk Tolerance: {client_data.get('risk_tolerance')}
    - Holdings Summary: {portfolio_summary}
    - Advisor Notes: {client_data.get('advisor_notes')}
    - Client's Latest Inquiry: "{client_data.get('latest_inquiry')}"

    **TASK:**
    Draft a reassuring, professional, and empathetic email. Acknowledge her specific concern about tech stock concentration and market news. Reinforce that her portfolio was built for her long-term goals. The main call to action should be to schedule a review call with her advisor.
    """

    print("\n" + "="*70)
    print("||" + " AI-Generated Draft for Advisor Review (Streaming...) ".center(64) + "||")
    print("="*70)

    try:
        stream = client.chat.completions.create(
            model="gpt-4-turbo-preview",
            messages=[{"role": "user", "content": prompt}],
            stream=True
        )
        
        for chunk in stream:
            print(chunk.choices[0].delta.content or "", end="")
        
        print("\n" + "="*70)

    except Exception as e:
        print(f"Error generating draft: {e}")

if __name__ == "__main__":
    print("#"*70)
    print("##" + " Lab 5: Advisor Co-pilot ".center(66) + "##")
    print("#"*70)
    
    client_profile = load_client_data('client_data.json')
    
    if client_profile:
        generate_advisor_draft(client_profile)

