import chromadb
from chromadb.config import Settings
import requests
from openai import OpenAI
import google.generativeai as genai # type: ignore
import json
from llamaapi import LlamaAPI # type: ignore
from dotenv import load_dotenv
import os

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
genai_api_key = os.getenv("GENAI_API_KEY")
llama_api_key = os.getenv("LLAMA_API_KEY")
client = OpenAI(api_key=openai_api_key)
genai.configure(api_key=genai_api_key)
llama = LlamaAPI(llama_api_key)

def initialize_chroma_db(persist_directory="./chroma_db"):
    client = chromadb.PersistentClient(path=persist_directory)
    return client

def retrieve_context(collection, query, top_k=10):
    query_embedding = generate_query_embedding(query)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["embeddings", "documents", "metadatas", "distances"]
    )
    documents = results.get("documents", [])
    flattened_documents = [doc for sublist in documents for doc in (sublist or []) if doc]
    return flattened_documents

def generate_query_embedding(query):
    response = client.embeddings.create(model="text-embedding-ada-002", input=query)
    return response.data[0].embedding

def generate_response_llama(context, query):
    try:
        api_request_json = {
            "model": "llama3.1-70b",
            "messages": [
                {"role": "system", "content": "You are an intelligent assistant that can answer questions based on provided context. The context is from emails, be sure to give satisfactory response"},
                {"role": "user", "content": f"Context: {context}\n\nQuestion: {query}"}
            ],
            "stream": False 
        }
        response = llama.run(api_request_json)
        return response.json().get("choices", [{}])[0].get("message", {}).get("content", "No response.")
    except Exception as e:
        return f"Error with Llama LLM: {e}"

def generate_response_gemini(context, query,):
    prompt = f"You are an intelligent assistant that can answer questions based on provided context. The context is from emails, be sure to give satisfactory response.\n\nContext:\n{context}\n\nQuestion:\n{query}\n\nAnswer:"
    model = genai.GenerativeModel("gemini-1.5-pro")
    response = model.generate_content(prompt)
    return response.text

def generate_response_gpt4(context, query):
    prompt = f"Context:\n{context}\n\nQuestion:\n{query}\n\nAnswer:"
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are an intelligent assistant that can answer questions based on provided context. The context is from emails, be sure to give satisfactory response"},
            {"role": "user", "content": prompt}
        ]
    )
    return (response.choices[0].message.content)

def append_response(output_file, prompt, response):
    with open(output_file, "a", encoding="utf-8") as f:
        f.write(f"Q: {prompt}\n")
        f.write(f"A: {response}\n\n")

def summarize_context(context):
    prompt = f"Summarize the following context in 850 words:\n\n{context}"
    summary = client.completions.create(
        model="gpt-3.5-turbo-instruct",
        prompt=prompt,
        max_tokens=850
    )
    return summary["choices"][0]["text"]

print("Initializing ChromaDB...")
chroma_client = initialize_chroma_db(persist_directory="./chroma_db")
collections = chroma_client.list_collections()
print(f"Available Collections: {[col.name for col in collections]}")
collection_name = "email_embeddings"
collection = chroma_client.get_collection(name=collection_name)
print(collection)

prompts = [
    "Search for deals from Jockey for black friday.",
    "How many job applications I have applied to till now",
    "Are there any pending returns for any orders?",
    "How much have I spend on electronics till now",
    "Which company sends me the most promotional emails",
    "How many credit cards do I own",
    "What are the best restaurants in San Jose?",
    "Explain quantum computing in simple terms.",
    "How does the stock market work?",
    "List all the subscriptions I am currently paying for.",
    "What is the total amount spent on food delivery this month?",
    "Which email contains the details of my last flight booking?",
    "Are there any overdue bills or payments in my emails?",
    "How many emails contain attachments in the last month?",
    "Search for bank statement of chase bank for the month of october",
    "How many emails have I received from Netflix?",
    "Do I have any emails with gift card codes or promo codes that I haven’t used yet?",
    "Are there any pending refunds for canceled orders?",
    "How much have I spent on food delivery apps this year?", #19
    "What is the most frequent discount percentage offered in my emails?",
    "How many credit card bills are due this month?",
    "Which bank sends me the most updates or alerts?",
    "How many credit cards have I applied for in the last six months?",
    "Did I missed any credit card payments?",
    "How much loans I have on me?",
    "Which investment platform sends me the most frequent updates?", #26
    "Which job portals send me the most recommendations?",
    "Do I have any pending tasks or follow-ups mentioned in my emails?",
    "How many companies have I applied to in the last month?",
    "Are there any emails about discounts on flights or hotels?",
    "How much have I spent on travel bookings this year?",
    "Give me the best laptop deal that anyone has sent me.",
    "What are the most common themes in my academic or research emails?"
    "What are some good productivity tips?",
    "How can I save more money every month?",
]

output_files = {
    "gemini": "./gemini_responses.txt",
    "gpt4": "./gpt4_responses.txt",
    "llama": "./llama_responses.txt"
}

for llm, file_path in output_files.items():
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(f"{llm.upper()} Responses\n{'=' * 20}\n\n")

for i, prompt in enumerate(prompts):
    print("_"*100)
    print(f"Processing prompt {i + 1}: {prompt}")
    retrieved_documents = retrieve_context(collection=collection, query=prompt)
    context = " ".join(retrieved_documents)
    # context = summarize_context(context)
    context = context[:850]

    print("Generating response with Gemini...")
    response_gemini = generate_response_gemini(context, prompt)
    append_response(output_files["gemini"], prompt, response_gemini)
    print(f"Response (Gemini):\n{response_gemini}\n")

    print("Generating response with GPT-4...")
    response_gpt4 = generate_response_gpt4(context, prompt)
    append_response(output_files["gpt4"], prompt, response_gpt4)
    print(f"Response (GPT-4):\n{response_gpt4}\n")
    
    print("Generating response with Llama...")
    response_llama = generate_response_llama(context, prompt)
    append_response(output_files["llama"], prompt, response_llama)
    print(f"Response (Llama):\n{response_llama}\n")
    
print("All prompts processed. Responses saved in respective files.")