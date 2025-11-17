import os
from openai import OpenAI
from dotenv import load_dotenv

# --- Setup ---
load_dotenv()
client = OpenAI()

def print_header(title):
    print("\n" + "#"*60)
    print(f"## {title.center(56)} ##")
    print("#"*60)

# --- Task A: Discriminative (Classification Only) ---
def classify_transaction(system_prompt, transaction_data):
    print_header("Task A: Discriminative Analysis")
    print(f"Analyzing Transaction: '{transaction_data}'\n")

    prompt = f"Please classify the following transaction based on your rules: {transaction_data}"
    
    print("--------------- AI Analysis (Classification) ---------------")
    response = client.chat.completions.create(
        model="gpt-4-turbo-preview",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]
    )
    print(response.choices[0].message.content)
    print("----------------------------------------------------------")

# --- Task B: Generative (Classification + Creation) ---
def generate_report(system_prompt, transaction_data):
    print_header("Task B: Generative Analysis")
    print(f"Analyzing Same Transaction: '{transaction_data}'\n")

    prompt = f"""
    Perform a two-step analysis on the following transaction: '{transaction_data}'

    Step 1: Classify the transaction's risk based on your internal rules.
    Step 2: Based on your classification, draft the first two paragraphs of an Internal Compliance Review Memo. The memo should be formal and objective, outlining the facts of the transaction for a senior compliance officer.
    """

    print("--------------- AI Generated Memo Draft ---------------")
    stream = client.chat.completions.create(
        model="gpt-4-turbo-preview",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ],
        stream=True
    )
    for chunk in stream:
        print(chunk.choices[0].delta.content or "", end="")
    print("\n----------------------------------------------------")


if __name__ == "__main__":
    try:
        with open('system_prompt.txt', 'r') as file:
            system_prompt_content = file.read()
        
        # Define the single transaction we will analyze
        transaction_to_analyze = "A client sends $25,000 to the registered crypto exchange 'CoinBase'."

        # Run both tasks on the same data
        classify_transaction(system_prompt_content, transaction_to_analyze)
        generate_report(system_prompt_content, transaction_to_analyze)

    except FileNotFoundError:
        print("Error: system_prompt.txt not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

