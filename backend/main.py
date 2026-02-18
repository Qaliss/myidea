from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import supabase
import json
import os
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="MyIdea")

url = os.getenv("SUPABASE_URL")
key = os.getenv("SUPABASE_KEY")

supabase_client = supabase.create_client(url, key)

class IdeaRequest(BaseModel):
    embedding: List[float]
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

    response = supabase_client.rpc(
        "match_papers",
        {
            "query_embedding": request.embedding,
            "match_count": request.top_k
        }
    ).execute()

    if response.data is None:
        return []

    return response.data


@app.get("/")
async def root():
    return {"message": "Welcome to the idea analyzing service"}
