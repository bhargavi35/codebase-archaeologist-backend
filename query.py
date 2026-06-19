import sys
import chromadb
from google import genai

# 1. Initialize the Google GenAI Client
ai_client = genai.Client()

# 2. Connect to our existing local ChromaDB database
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_collection(name="pdf_documents")

def ask_pdf(question: str):
    # Step A: Convert the user's question into a math vector
    # We MUST use the exact same embedding model we used for the text chunks!
    response = ai_client.models.embed_content(
        model="gemini-embedding-2",
        contents=question
    )
    question_vector = response.embeddings[0].values
    
    # Step B: Query ChromaDB to find the top 3 most relevant chunks
    results = collection.query(
        query_embeddings=[question_vector],
        n_results=3
    )
    
    # Extract the text chunks found by our database
    relevant_chunks = results['documents'][0]
    
    # Step C: Combine the chunks into a unified context string
    context = "\n---\n".join(relevant_chunks)
    
    # Step D: Construct a strict system prompt instructing Gemini to behave
    system_instruction = (
        "You are an expert AI assistant reading a document. "
        "Answer the user's question using ONLY the provided context below. "
        "If the answer cannot be found within the context, say 'I cannot find the answer in the provided document.' "
        "Do not make up facts."
    )
    
    user_prompt = f"Context:\n{context}\n\nQuestion: {question}"
    
    print("\n🔍 Fetching answer from Gemini...")
    
    # Step E: Call Gemini to generate the final answer
    # Using 'gemini-2.5-flash' because it's exceptionally fast and optimized for RAG tasks
    chat_response = ai_client.models.generate_content(
        model="gemini-2.5-flash",
        contents=user_prompt,
        config={"system_instruction": system_instruction}
    )
    
    print("\n🤖 AI Answer:")
    print(chat_response.text)
    print("-" * 40)

if __name__ == "__main__":
    # This allows you to pass a question via your terminal command
    if len(sys.argv) > 1:
        user_query = " ".join(sys.argv[1:])
        ask_pdf(user_query)
    else:
        # Fallback test question if you run it plain
        ask_pdf("What is the main topic discussed in this document?")