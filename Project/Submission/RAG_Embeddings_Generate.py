import os
import re
from openai import OpenAI
import chromadb
from dotenv import load_dotenv
import os

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def initialize_chroma_db(persist_directory="./chroma_db"):
    client = chromadb.PersistentClient(path=persist_directory)
    return client

def create_collection(client, collection_name="email_embeddings"):
    return client.get_or_create_collection(name=collection_name)

def generate_openai_embedding(email_text):
    try:
        response = client.embeddings.create(model="text-embedding-ada-002", input=email_text)
        return response.data[0].embedding
    except Exception as e:
        print(f"Error generating embedding: {e}")
        return None

def read_email_files(data_folder):
    email_texts = []
    email_files = []
    for file_name in os.listdir(data_folder):
        file_path = os.path.join(data_folder, file_name)
        if os.path.isfile(file_path) and file_name.endswith(".txt"):
            with open(file_path, "r", encoding="utf-8") as file:
                raw_text = file.read()
                cleaned_text = clean_email_body(raw_text)
                email_texts.append(cleaned_text)
                email_files.append(file_name)
    return email_texts, email_files

def clean_email_body(email_text):
    parts = email_text.split("\n\n", 1)
    headers = parts[0] 
    body = parts[1] if len(parts) > 1 else ""
    body = re.sub(r"\s+", " ", body)
    body = re.sub(r"[^a-zA-Z0-9.,!?:/\-_@'\" ]+", " ", body)
    body = body.strip()

    cleaned_email = f"{headers}\n\n{body}"
    return cleaned_email

def extract_metadata(email_text):
    subject = re.search(r"^Subject: (.+)", email_text, re.MULTILINE)
    sender = re.search(r"^From: (.+)", email_text, re.MULTILINE)
    recipient = re.search(r"^To: (.+)", email_text, re.MULTILINE)
    date = re.search(r"^Date: (.+)", email_text, re.MULTILINE)

    metadata = {
        "subject": subject.group(1) if subject else "No Subject",
        "from": sender.group(1) if sender else "Unknown Sender",
        "to": recipient.group(1) if recipient else "Unknown Recipient",
        "date": date.group(1) if date else "Unknown Date"
    }
    return metadata

def add_email_to_collection(collection, email_text, email_id, metadata):
    try:
        embedding = generate_openai_embedding(email_text)
        if embedding is None:
            print(f"Skipping email {email_id} due to embedding generation failure.")
            return
        collection.add(
            embeddings=[embedding],
            ids=[email_id],
            metadatas=[metadata],
            documents=[email_text]
        )
        print(f"Email {email_id} successfully added to ChromaDB.")
    except Exception as e:
        print(f"Failed to add email {email_id}: {e}")


data_folder = "./data"
persist_directory = "./chroma_db"
chroma_client = initialize_chroma_db(persist_directory=persist_directory)
print("Chroma client created")
collection = create_collection(chroma_client, collection_name="email_embeddings")
print("Collection Created")

email_texts, email_files = read_email_files(data_folder)
for i, email_text in enumerate(email_texts):
    email_id = os.path.splitext(email_files[i])[0] 
    metadata = extract_metadata(email_text) 
    print(f"Processing email {email_id}...")
    add_email_to_collection(collection, email_text, email_id, metadata)

print("All emails successfully processed and stored in ChromaDB.")