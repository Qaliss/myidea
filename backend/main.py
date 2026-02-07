from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import supabase
from sentence_transformers import SentenceTransformer
import numpy as np
import json
import os
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="MyIdea")

model = SentenceTransformer("all-MiniLM-L6-v2")

url = os.getenv("SUPABASE_URL")
key = os.getenv("SUPABASE_KEY")

supabase_client = supabase.create_client(url, key)

class IdeaRequest(BaseModel):
    idea: str
    top_k: int = 10

class PaperResponse(BaseModel):
    title: str
    abstract: str
    authors: List[str]
    categories: List[str]
    published_date: str
    score: float

@app.post("/analyze_idea", response_model=List[PaperResponse])
def analyze_idea(request: IdeaRequest):
    # 1️⃣ Embed the idea
    idea_embedding = model.encode(request.idea).tolist()

    # 2️⃣ Call Supabase RPC (pgvector similarity)
    response = supabase_client.rpc(
        "match_papers",
        {
            "query_embedding": idea_embedding,
            "match_count": request.top_k
        }
    ).execute()

    if response.data is None:
        return []

    return response.data
