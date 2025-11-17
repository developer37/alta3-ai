import os
from openai import OpenAI
from dotenv import load_dotenv

# --- Setup ---
load_dotenv()
client = OpenAI()

def load_documents(folder):
    docs = {}
    for filename in os.listdir(folder):
        if filename.endswith(".txt"):
            with open(os.path.join(folder, filename), 'r') as file:
                docs[filename] = file.read()
    return docs

def find_best_document_with_llm(query, doc_names):
    """Uses a cheap, fast LLM to act as a document router."""
    doc_list = "\n".join(doc_names)
    
    prompt = f"""
    You are a document routing assistant. Your job is to determine which of the following documents is most relevant to the user's question.
    Respond with ONLY the filename of the best document.

    Available Documents:
    {doc_list}

    User's Question:
    "{query}"
    """
    
    response = client.chat.completions.create(
        model="gpt-3.5-turbo", # Use a cheap, fast model for this routing task
        messages=[{"role": "user", "content": prompt}]
    )
    
    # The response should be just the filename
    return response.choices[0].message.content.strip()

def main_chat_loop(documents):
    print("\n" + "="*60)
    print("      Policy Q&A Bot (LLM Router RAG) - Type 'quit' to exit")
    print("="*60)
    
    doc_filenames = list(documents.keys())

    while True:
        user_question = input("\n> Your Question: ")
        if user_question.lower() == 'quit':
            break
        
        print(f"\n[System] Asking LLM Router to find the best document...")
        retrieved_doc_name = find_best_document_with_llm(user_question, doc_filenames)
        
        if retrieved_doc_name not in documents:
            print(f"[System] Router failed or returned an invalid document name. Please try rephrasing.")
            continue

        retrieved_doc_text = documents[retrieved_doc_name]
        print(f"[System] Router selected: '{retrieved_doc_name}'. Answering question...")
        
        final_prompt = f"Answer the user's question based ONLY on the following context:\n\nCONTEXT: --- {retrieved_doc_text} ---\n\nQUESTION: {user_question}"
        
        stream = client.chat.completions.create(
            model="gpt-4-turbo-preview", # Use the powerful model for the final answer
            messages=[{"role": "user", "content": final_prompt}],
            stream=True
        )
        
        print("\n" + "-"*15 + " AI Answer " + "-"*15)
        for chunk in stream:
            print(chunk.choices[0].delta.content or "", end="")
        print("\n" + "-"*41)

if __name__ == "__main__":
    knowledge_base = load_documents("knowledge_base")
    main_chat_loop(knowledge_base)
    print("\nExiting bot. Goodbye!")

