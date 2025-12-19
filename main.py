import os
from typing import Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import chromadb
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

# --- Pydantic Model ---
class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    answer: str
    retrieved_count: int

# --- RAG Logic ---
class RAGService:
    def __init__(self):
        # 1. Setup Gemini
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        self.llm = genai.GenerativeModel('models/gemini-2.5-flash') 

        # 2. Setup Embedding
        print("Loading embedding model...")
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')

        # 3. Connect to Chroma Cloud
        print("Connecting to Chroma Cloud...")
        self.client = chromadb.CloudClient(
            api_key=os.getenv("CHROMA_API_KEY"),
            tenant=os.getenv("CHROMA_TENANT"),
            database=os.getenv("CHROMA_DATABASE")
        )
        self.collection = self.client.get_collection(
            name=os.getenv("CHROMA_COLLECTION", "investigation_docs")
        )

    def process_query(self, user_query: str, case_id: str = None) -> str:
        print(f"Processing: {user_query} | Case ID: {case_id}")
        
        # A. Embed query
        query_vector = self.embedder.encode([user_query]).tolist()
        
        # B. Define Filter
        where_filter = {"case_id": case_id} if case_id else None

        # C. Retrieve
        results = self.collection.query(
            query_embeddings=query_vector,
            n_results=5,
            where=where_filter
        )
        
        retrieved_docs = results['documents'][0] if results['documents'] else []

        if not retrieved_docs:
            if case_id:
                return f"No documents found specifically for Case ID: {case_id}."
            return "No relevant information found in the database."

        # D. Generate
        context_text = "\n\n".join(retrieved_docs)
        prompt = f"""
        You are an intelligent legal assistant. Answer the question based ONLY on the context below.
        
        CONTEXT (Case {case_id if case_id else 'General'}):
        {context_text}
        
        QUESTION: 
        {user_query}
        """
        
        response = self.llm.generate_content(prompt)
        return response.text

# --- FastAPI App ---
app = FastAPI(title="RAG Retrieval API")
rag_service = None

@app.on_event("startup")
async def startup_event():
    global rag_service
    rag_service = RAGService()
    print("RAG Service Initialized.")

@app.post("/query/{case_id}", response_model=QueryResponse)
async def query_case(case_id: str, request: QueryRequest):
    if not rag_service:
        raise HTTPException(status_code=503, detail="Service starting up...")
    try:
        answer = rag_service.process_query(request.query, case_id)
        return QueryResponse(answer=answer, retrieved_count=5)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)