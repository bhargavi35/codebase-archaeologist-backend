import os
import re
import io
import zipfile
import httpx
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Dict
from pypdf import PdfReader
import chromadb
from google import genai
from google.genai.errors import ClientError

app = FastAPI(title="Codebase Archaeologist Production Engine")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

render_api_key = os.environ.get("GEMINI_API_KEY", "")
ai_client = genai.Client(api_key=render_api_key)
# chroma_client = chromadb.PersistentClient(path="./chroma_db")
chroma_client = chromadb.EphemeralClient()
collection = chroma_client.get_or_create_collection(name="unified_knowledge_base")

MAX_INDEX_LIMIT = 15  # Protective cap to survive free tier quota window limits

class ChatMessage(BaseModel):
    role: str
    text: str

class UnifiedChatRequest(BaseModel):
    question: str
    history: List[ChatMessage]

class RepoImportRequest(BaseModel):
    repo_url: str

def extract_imports(file_content: str):
    # Enhanced matching tracking relative paths, aliased modules, and standard paths
    pattern = r"from\s+['\"]([^'\"]+)['\"]|import\s+['\"]([^'\"]+)['\"]"
    matches = re.findall(pattern, file_content)
    return [m[0] or m[1] for m in matches if m[0] or m[1]]

def chunk_pdf_text(pdf_file, chunk_size=600, chunk_overlap=120):
    reader = PdfReader(pdf_file)
    full_text = ""
    for page in reader.pages:
        text = page.extract_text()
        if text: full_text += text + "\n"
    chunks = []
    start = 0
    while start < len(full_text):
        end = start + chunk_size
        chunks.append(full_text[start:end])
        start += (chunk_size - chunk_overlap)
    return chunks

@app.post("/github/import")
async def import_github_repository(payload: RepoImportRequest):
    url = payload.repo_url.strip().rstrip("/")
    if not url.startswith("https://github.com/"):
        raise HTTPException(status_code=400, detail="Invalid GitHub Repository URL address template.")
    
    # FIX: Bypass api.github.com completely to avoid 404 blocks on anonymous lookups!
    # This maps cleanly to the default 'main' branch zip archive structure.
    clean_url = f"{url}/archive/refs/heads/main.zip"
    
    try:
        async with httpx.AsyncClient() as client:
            # We add a standard browser User-Agent header so GitHub doesn't drop the connection
            headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
            response = await client.get(clean_url, headers=headers, follow_redirects=True)
            
            # FALLBACK: If your repo defaults to using 'master' instead of 'main', try master archive
            if response.status_code == 404:
                fallback_url = f"{url}/archive/refs/heads/master.zip"
                response = await client.get(fallback_url, headers=headers, follow_redirects=True)
                
            if response.status_code != 200:
                raise HTTPException(status_code=404, detail="Target repository repository source zip archive not found.")
        
        # Flush previous session records for clean layout generation
        global collection
        chroma_client.delete_collection(name="unified_knowledge_base")
        collection = chroma_client.get_or_create_collection(name="unified_knowledge_base")
        
        zip_bytes = io.BytesIO(response.content)
        indexed_count = 0
        
        with zipfile.ZipFile(zip_bytes) as archive:
            for file_info in archive.infolist():
                if indexed_count >= MAX_INDEX_LIMIT:
                    break
                
                filename = file_info.filename
                if any(x in filename for x in ["node_modules", ".next", "package", "config", "assets", ".git"]):
                    continue
                if not filename.endswith(('.ts', '.tsx', '.js', '.jsx')):
                    continue
                    
                content = archive.read(file_info).decode("utf-8", errors="ignore")
                if not content.strip() or len(content) < 40:
                    continue
                
                display_name = "/".join(filename.split("/")[1:])
                dependencies = extract_imports(content)
                
                v_res = ai_client.models.embed_content(model="gemini-embedding-2", contents=content)
                vector = v_res.embeddings[0].values
                
                collection.add(
                    documents=[content],
                    embeddings=[vector],
                    metadatas=[{
                        "file_name": display_name,
                        "file_category": "code",
                        "file_type": os.path.splitext(display_name)[1],
                        "imports": ",".join(dependencies)
                    }],
                    ids=[display_name]
                )
                indexed_count += 1
                
        return {"success": True, "message": f"Successfully parsed and mapped {indexed_count} architectural modules layout."}
    except ClientError as ce:
        if getattr(ce, "code", None) == 429:
            raise HTTPException(status_code=429, detail="Gemini Rate Limit reached. Please wait a moment.")
        raise HTTPException(status_code=500, detail=str(ce))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/assets")
async def get_uploaded_assets():
    try:
        results = collection.get(include=["metadatas"])
        metadatas = results.get("metadatas", [])
        seen_files = {}
        for meta in metadatas:
            name = meta.get("file_name")
            if name and name not in seen_files:
                seen_files[name] = {
                    "name": name,
                    "category": meta.get("file_category"),
                    "type": meta.get("file_type")
                }
        return {"assets": list(seen_files.values())}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/clear")
async def clear_database():
    try:
        global collection
        chroma_client.delete_collection(name="unified_knowledge_base")
        collection = chroma_client.get_or_create_collection(name="unified_knowledge_base")
        return {"success": True, "message": "Database successfully flushed."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload")
async def upload_asset(file: UploadFile = File(...)):
    if file.filename.endswith(".pdf"):
        # Process standard chunk array logic
        try:
            chunks = chunk_pdf_text(file.file)
            documents, embeddings, metadatas, ids = [], [], [], []
            for index, chunk in enumerate(chunks):
                response = ai_client.models.embed_content(model="gemini-embedding-2", contents=chunk)
                vector = response.embeddings[0].values
                documents.append(chunk)
                embeddings.append(vector)
                ids.append(f"{file.filename}_chunk_{index}")
                metadatas.append({"file_name": file.filename, "file_category": "document", "file_type": ".pdf"})
            collection.add(documents=documents, embeddings=embeddings, metadatas=metadatas, ids=ids)
            return {"success": True, "type": "document"}
        except Exception as e: raise HTTPException(status_code=500, detail=str(e))
    else:
        # Standard code element fallback
        try:
            content_bytes = await file.read()
            content = content_bytes.decode("utf-8")
            dependencies = extract_imports(content)
            res = ai_client.models.embed_content(model="gemini-embedding-2", contents=content)
            vector = res.embeddings[0].values
            collection.add(documents=[content], embeddings=[vector], metadatas=[{"file_name": file.filename, "file_category": "code", "file_type": ".tsx", "imports": ",".join(dependencies)}], ids=[file.filename])
            return {"success": True, "type": "code"}
        except Exception as e: raise HTTPException(status_code=500, detail=str(e))

@app.get("/codebase/map")
async def get_codebase_map():
    try:
        results = collection.get(include=["metadatas"])
        metadatas = results.get("metadatas", [])
        ids = results.get("ids", [])
        nodes, edges = [], []
        code_node_count = 0
        
        for idx, file_id in enumerate(ids):
            meta = metadatas[idx]
            if meta.get("file_category") == "code":
                nodes.append({
                    "id": file_id,
                    "type": "default",
                    "data": { "label": meta.get("file_name", file_id).split("/")[-1] },
                    "position": { "x": (code_node_count % 3) * 240, "y": (code_node_count // 3) * 150 },
                    "style": { "background": "#1e1b4b", "color": "#f8fafc", "border": "1px solid #4f46e5", "borderRadius": "8px" }
                })
                code_node_count += 1
                
                raw_imports = meta.get("imports", "")
                if raw_imports:
                    for imp in raw_imports.split(","):
                        for target_id in ids:
                            target_base = target_id.split(".")[0].split("/")[-1]
                            if target_base in imp and file_id != target_id:
                                edges.append({
                                    "id": f"edge_{file_id}_to_{target_id}",
                                    "source": file_id,
                                    "target": target_id,
                                    "animated": True,
                                    "style": { "stroke": "#6366f1" }
                                })
        return {"nodes": nodes, "edges": edges}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat")
async def unified_hybrid_chat(payload: UnifiedChatRequest):
    try:
        def response_streamer():
            try:
                response = ai_client.models.embed_content(model="gemini-embedding-2", contents=payload.question)
                question_vector = response.embeddings[0].values
                results = collection.query(query_embeddings=[question_vector], n_results=4)
                relevant_blocks = results['documents'][0]
                context = "\n\n--- UNIFIED CONTEXT BLOCK ---\n\n".join(relevant_blocks) if relevant_blocks else "No matching contexts found."
                
                system_instruction = (
                    "You are an expert AI software architect. Analyze codebase questions mapping execution flows. "
                    "Provide clean answers based strictly on the context chunks."
                )
                formatted_history = []
                for msg in payload.history:
                    if msg.text: formatted_history.append({"role": msg.role, "parts": [{"text": msg.text}]})
                        
                chat_session = ai_client.chats.create(model="gemini-2.5-flash", history=formatted_history, config={"system_instruction": system_instruction})
                stream = chat_session.send_message_stream(f"Context:\n{context}\n\nQuestion: {payload.question}")
                for chunk in stream:
                    if chunk.text: yield f"data: {chunk.text}\n\n"
            except ClientError as ce:
                if getattr(ce, "code", None) == 429: yield "data: ERROR_QUOTA_EXHAUSTED\n\n"
                else: yield "data: ERROR_STREAM_FAILED\n\n"
            except Exception: yield "data: ERROR_STREAM_FAILED\n\n"

        return StreamingResponse(response_streamer(), headers={"Connection": "keep-alive", "Cache-Control": "no-cache", "Content-Type": "text/event-stream"})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))