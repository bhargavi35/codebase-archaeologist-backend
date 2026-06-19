import os
from pypdf import PdfReader
import chromadb
from google import genai

# 1. Initialize the Google GenAI Client (Automatically picks up GEMINI_API_KEY)
ai_client = genai.Client()

# 2. Setup Persistent Local ChromaDB Storage
# This creates a folder named './chroma_db' so your vectors save to your disk!
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="pdf_documents")

def extract_and_chunk_pdf(pdf_path, chunk_size=500, chunk_overlap=100):
    """Reads a PDF file, extracts text, and cuts it into overlapping chunks."""
    reader = PdfReader(pdf_path)
    full_text = ""
    
    # Extract text from every page
    for page in reader.pages:
        text = page.extract_text()
        if text:
            full_text += text + "\n"
            
    # Chunking logic using sliding window mechanics
    chunks = []
    start = 0
    while start < len(full_text):
        end = start + chunk_size
        chunk = full_text[start:end]
        chunks.append(chunk)
        start += (chunk_size - chunk_overlap) # slide forward accounting for overlap
        
    return chunks

def ingest_pdf(pdf_path):
    print(f"🔄 Processing: {pdf_path}...")
    chunks = extract_and_chunk_pdf(pdf_path)
    print(f"✂️ Sliced PDF into {len(chunks)} text chunks.")
    
    documents = []
    embeddings = []
    ids = []
    
    # Loop over every text chunk and calculate its mathematical representation
    for index, chunk in enumerate(chunks):
        # Generate dense vector embedding from Gemini
        response = ai_client.models.embed_content(
            model="gemini-embedding-2",
            contents=chunk
        )
        # Extract the array of numbers (the vector)
        vector = response.embeddings[0].values
        
        documents.append(chunk)
        embeddings.append(vector)
        ids.append(f"chunk_{index}")
        
    # Upsert data into our local vector database
    collection.add(
        documents=documents,
        embeddings=embeddings,
        ids=ids
    )
    print("✅ Ingestion complete! Vectors successfully saved locally to disk.")

if __name__ == "__main__":
    # Drop any sample PDF file in this directory and name it "sample.pdf" to test it!
    if os.path.exists("sample.pdf"):
        ingest_pdf("sample.pdf")
    else:
        print("❌ Please put a 'sample.pdf' in this folder to run a local test.")