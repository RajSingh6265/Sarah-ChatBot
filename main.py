from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import requests
import PyPDF2
import io
from typing import List, Dict, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import json
import os

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Hugging Face API configuration
API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.3"
headers = {"Authorization": f"Bearer {os.getenv('HUGGINGFACE_API_KEY')}"}

# Initialize sentence transformer for embeddings
embedder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Store PDF content and embeddings
pdf_content = []
pdf_embeddings = None

class Message(BaseModel):
    content: str
    use_pdf: bool = True

def extract_text_from_pdf(pdf_file: bytes) -> str:
    """Extract text from uploaded PDF file"""
    pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_file))
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() + "\n"
    return text

def split_into_chunks(text: str, chunk_size: int = 1000) -> List[str]:
    """Split text into chunks of approximately equal size"""
    words = text.split()
    chunks = []
    current_chunk = []
    current_size = 0
    
    for word in words:
        current_chunk.append(word)
        current_size += len(word) + 1
        
        if current_size >= chunk_size:
            chunks.append(" ".join(current_chunk))
            current_chunk = []
            current_size = 0
            
    if current_chunk:
        chunks.append(" ".join(current_chunk))
        
    return chunks

def get_relevant_chunks(query: str, top_k: int = 3) -> List[str]:
    """Get most relevant chunks for the query using cosine similarity"""
    if pdf_embeddings is None or not pdf_content:
        return []
        
    query_embedding = embedder.encode([query])[0]
    similarities = cosine_similarity([query_embedding], pdf_embeddings)[0]
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    
    return [pdf_content[i] for i in top_indices if similarities[i] > 0.3]

def format_prompt(message: str, context: Optional[List[str]] = None) -> str:
    """Format the prompt for Mistral model"""
    if context:
        context_str = "\n".join(context)
        return f"""<s>[INST] Using the following context, answer the question. If the question cannot be answered from the context, say so and provide a general response.

Context:
{context_str}

Question: {message} [/INST]</s>"""
    else:
        return f"""<s>[INST] {message} [/INST]</s>"""

@app.post("/upload-pdf")
async def upload_pdf(file: UploadFile = File(...)):
    try:
        global pdf_content, pdf_embeddings
        
        # Read and extract text from PDF
        pdf_bytes = await file.read()
        text = extract_text_from_pdf(pdf_bytes)
        
        # Split text into chunks
        pdf_content = split_into_chunks(text)
        
        # Generate embeddings
        pdf_embeddings = embedder.encode(pdf_content)
        
        return {"message": "PDF processed successfully", "chunks": len(pdf_content)}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat")
async def chat(message: Message):
    try:
        relevant_chunks = []
        if message.use_pdf and pdf_embeddings is not None:
            relevant_chunks = get_relevant_chunks(message.content)
        
        # Format the prompt with or without context
        formatted_prompt = format_prompt(message.content, relevant_chunks if relevant_chunks else None)
        
        # Make request to Hugging Face API
        payload = {
            "inputs": formatted_prompt,
            "parameters": {
                "max_new_tokens": 512,
                "temperature": 0.7,
                "top_p": 0.95,
                "do_sample": True,
                "return_full_text": False
            }
        }
        
        response = requests.post(API_URL, headers=headers, json=payload)
        
        if response.status_code != 200:
            raise HTTPException(status_code=response.status_code, detail="Model API request failed")
            
        # Extract the generated text from response
        response_data = response.json()
        if isinstance(response_data, list) and len(response_data) > 0:
            generated_text = response_data[0].get('generated_text', '')
            generated_text = generated_text.strip()
        else:
            generated_text = "I apologize, but I couldn't generate a proper response."
            
        return {
            "response": generated_text,
            "used_pdf": bool(relevant_chunks),
            "chunks_used": len(relevant_chunks)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/", response_class=HTMLResponse)
async def root():
    with open("templates/index.html", "r") as f:
        return f.read()

