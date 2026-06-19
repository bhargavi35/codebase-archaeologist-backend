# 🐍 Codebase Archaeologist Production Engine — Backend API

A high-performance, asynchronous FastAPI orchestration layer designed to drive semantic discovery, map software structural dependencies, and execute real-time code vector transformations.

📂 **Live API Interactive Docs:** [https://codebase-archaeologist-backend.onrender.com/docs](https://codebase-archaeologist-backend.onrender.com/docs)

---

## ✨ Key Capabilities

* **🌐 Memory-Buffered GitHub Ingestion:** Securely clones public GitHub ZIP archives directly into short-lived system memory, bypassing local disk clutter, to parse and map source file modules on the fly.
* **🧠 Free-Tier Cloud Optimized Indexing:** Utilizes a high-speed `chromadb` Ephemeral Memory Client to instantiate in-RAM data collections, bypassing the need for paid cloud disk space while ensuring zero-leak data flushes.
* **🔮 AST-Inspired Import Extractor:** Employs advanced regex parsing arrays to scan incoming code files, pull relative/aliased paths, and construct an interactive network link matrix.
* **📡 Real-Time SSE Response Streaming:** Implements Server-Sent Events (SSE) via FastAPI's `StreamingResponse` to push real-time, token-by-token answer responses directly to the client dashboard.
* **🛡️ Defensive Quota Boundaries:** Enforces protective file capping constraints (maximum 15 concurrent source files mapped) and traps rate-limiting failures to pass down unified `ERROR_QUOTA_EXHAUSTED` tokens gracefully.

---

## 🛠️ Technical Stack & Architecture

* **Language/Framework:** Python 3.11+, FastAPI, Uvicorn
* **AI & Embeddings:** Google GenAI SDK (`gemini-embedding-2`, `gemini-2.5-flash`)
* **Vector Database:** ChromaDB (In-Memory Ephemeral Client mapping configuration)
* **HTTP Infrastructure:** HTTPX (Asynchronous network connection routing)

---

## 🚀 Local Deployment & API Testing

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/bhargavi35/codebase-archaeologist-backend.git](https://github.com/bhargavi35/codebase-archaeologist-backend.git)
   cd codebase-archaeologist-backend
Set up a virtual environment:

Bash
python -m venv venv
# On Windows:
.\venv\Scripts\activate
Install production dependencies:

Bash
pip install -r requirements.txt
Inject Your AI Key Secret:
Configure a .env file or export the key variable directly to your environment:

Bash
export GEMINI_API_KEY="your_google_ai_studio_key_here"
Fire up the Uvicorn worker:

Bash
uvicorn main:app --reload
Access the live interactive Swagger UI testing playground at http://127.0.0.1:8000/docs.