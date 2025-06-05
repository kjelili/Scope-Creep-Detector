# scope_rag_checker.py
import openai
import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import re
import hashlib

# Module-level cache for storing embeddings
EMBEDDING_CACHE = {}

# Risk normalization function
def normalize_risk(risk_str):
    """Standardize risk level casing and values"""
    if not risk_str:
        return "unknown"
        
    risk = risk_str.strip().lower()
    if risk in ["high", "extreme", "critical"]:
        return "high"
    elif risk in ["medium", "mod"]:
        return "moderate"
    elif risk in ["low", "minor"]:
        return "low"
    return risk

def get_embedding(text, api_key, model="text-embedding-3-small"): # Changed model to text-embedding-3-small
    """Generate or retrieve cached embedding for text"""
    text_hash = hashlib.sha256(text.encode()).hexdigest()
    
    if text_hash in EMBEDDING_CACHE:
        return EMBEDDING_CACHE[text_hash]
    
    client = openai.OpenAI(api_key=api_key)
    response = client.embeddings.create(
        input=[text],
        model=model
    )
    embedding = response.data[0].embedding
    EMBEDDING_CACHE[text_hash] = embedding
    return embedding

def chunk_text(text, chunk_size=500, overlap=55): # Adjusted overlap
    """Split text into overlapping chunks with sentence awareness"""
    text = re.sub(r'\s+', ' ', text).strip()
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        if len(current_chunk) + len(sentence) <= chunk_size:
            current_chunk += sentence + " "
        else:
            chunks.append(current_chunk.strip())
            # Ensure overlap doesn't exceed current_chunk length
            current_chunk = current_chunk[-min(len(current_chunk), overlap):] + " " + sentence + " "
    
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks

def retrieve_relevant_chunks(query, documents, api_key, top_k=3):
    """Retrieve top_k most relevant document chunks using cosine similarity"""
    query_embedding = np.array(get_embedding(query, api_key)).reshape(1, -1)
    
    doc_embeddings = []
    for doc in documents:
        doc_embeddings.append(get_embedding(doc, api_key))
    
    similarities = cosine_similarity(query_embedding, doc_embeddings)[0]
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    
    return [(documents[i], similarities[i]) for i in top_indices]

def check_scope_creep_with_rag(email, scope_text, api_key):
    try:
        client = openai.OpenAI(api_key=api_key)
        scope_chunks = chunk_text(scope_text)
        relevant_chunks = retrieve_relevant_chunks(email, scope_chunks, api_key)
        
        context = "\n\n".join([
            f"Scope Section (Relevance: {score:.2f}):\n{chunk}"
            for chunk, score in relevant_chunks
        ])
        
        messages = [
            {
                "role": "system",
                "content": (
                    "You are an AI project management assistant analyzing for scope creep. "
                    "Your task is to analyze the EMAIL content against the PROJECT SCOPE SECTIONS. "
                    "Provide a JSON response with the following keys:\n"
                    "1. `scope_creep`: (yes/no) - Whether scope creep is detected.\n"
                    "2. `justification`: (string) - Specific differences between email and scope.\n"
                    "3. `suggestion`: (string) - Suggested response strategy for the project manager.\n"
                    "4. `risk_level`: (Low/Moderate/High/Extreme) - Assessment of the risk.\n"
                    "5. `reference_scope_line`: (string) - Exact lines or sections from the scope that are relevant.\n"
                    "6. `impact_analysis`: (string) - Estimated cost/timeline impacts if the new request is accepted."
                )
            },
            {
                "role": "user",
                "content": (
                    "PROJECT SCOPE SECTIONS:\n"
                    "----------------------\n"
                    f"{context}\n\n"
                    "EMAIL CONTENT:\n"
                    "-------------\n"
                    f"{email}"
                )
            }
        ]
        
        response = client.chat.completions.create(
            model="gpt-4o", # Changed to gpt-4o for potentially better performance
            messages=messages,
            temperature=0.2,
            response_format={"type": "json_object"}
        )
        
        # Parse and normalize the response
        result = json.loads(response.choices[0].message.content)
        result["risk_level"] = normalize_risk(result.get("risk_level", "Unknown"))
        return result
    
    except Exception as e:
        return {
            "scope_creep": "error",
            "justification": f"An error occurred during AI analysis: {str(e)}",
            "suggestion": "Check logs for details and ensure API key is valid.",
            "risk_level": normalize_risk("Unknown"),
            "reference_scope_line": "none",
            "impact_analysis": "unknown"
        }