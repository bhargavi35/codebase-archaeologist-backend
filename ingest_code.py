from dotenv import load_dotenv
load_dotenv() # This automatically injects your .env file into Python's environment variables

import os
import re
import chromadb
from google import genai

# Initialize clients
ai_client = genai.Client()
chroma_client = chromadb.PersistentClient(path="./chroma_db")

# We create a brand new collection specifically for code assets
collection = chroma_client.get_or_create_collection(name="unified_knowledge_base")

def extract_imports(file_content: str):
    """Uses regex to extract import paths from TypeScript/JavaScript files."""
    # Matches: import X from './Component' or import { Y } from 'dependency'
    pattern = r"from\s+['\"]([^'\"]+)['\"]"
    return re.findall(pattern, file_content)

def scan_and_index_codebase(target_directory: str):
    print(f"🕵️‍♂️ Commencing codebase archaeology in: {target_directory}...")
    
    documents = []
    embeddings = []
    metadatas = []
    ids = []
    
    # Recursively traverse directory trees while ignoring node_modules and builds
    exclude_dirs = { "node_modules", ".next", "dist", "build", "venv", ".git" }
    supported_extensions = { ".ts", ".tsx", ".js", ".jsx" }

    for root, dirs, files in os.walk(target_directory):
        dirs[:] = [d for d in dirs if d not in exclude_dirs]
        
        for file in files:
            ext = os.path.splitext(file)[1]
            if ext in supported_extensions:
                file_path = os.path.join(root, file)
                relative_path = os.path.relpath(file_path, target_directory)
                
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        content = f.read()
                    
                    if not content.strip():
                        continue
                        
                    # Extract dependency links
                    dependencies = extract_imports(content)
                    
                    # Generate embedding vector for the file's code content
                    response = ai_client.models.embed_content(
                        model="gemini-embedding-2",
                        contents=content
                    )
                    vector = response.embeddings[0].values
                    
                    documents.append(content)
                    embeddings.append(vector)
                    ids.append(relative_path)
                    
                    # Core metadata mapping for graph layout building
                    metadatas.append({
                        "file_path": relative_path,
                        "file_name": file,
                        "file_type": ext,
                        "imports": ",".join(dependencies) # Stored as comma-separated string for ChromaDB
                    })
                    print(f"✅ Indexed: {relative_path} (Found {len(dependencies)} imports)")
                    
                except Exception as e:
                    print(f"⚠️ Failed to index file {relative_path}: {str(e)}")

    if documents:
        # Batch insert structural vectors into ChromaDB
        collection.add(
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids
        )
        print(f"\n🎉 Archaeology Complete! Indexed {len(documents)} source files into the vector mapping index.")
    else:
        print("❌ No matching source files found to index.")

if __name__ == "__main__":
    # Test it out directly! 
    # Create a dummy folder called 'sample_code' or pass your current frontend folder path
    target_folder = "./sample_code"
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)
        # Create a quick dummy file inside to test the parser execution
        with open(os.path.join(target_folder, "Button.tsx"), "w") as f:
            f.write("import React from 'react';\nexport const Button = () => <button>Click me</button>;")
            
    scan_and_index_codebase(target_folder)