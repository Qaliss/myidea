import arxiv
import sentence_transformers
from supabase import create_client
from tqdm import tqdm
import os

SUPABASE_URL = "https://uuhkmxqpvfcrpuzkgjdg.supabase.co"
SUPABASE_KEY = "sb_publishable_FcrQ6X7Si84TpSFQu3sp9A_GtSkfX5G"

MAX_RESULTS_PER_CATEGORY = 8000

CATEGORIES = [
    # "cs.AI", "cs.LG", "cs.CL", "cs.RO", "cs.CV", "cs.NE",
    # "eess.SP", "eess.SY",
    # "physics.app-ph", "physics.data-an",
    # "q-bio.BM", "q-bio.NC",
    #Need to continue the downloading from q-bio.NC
    # "q-bio.NC",
    # "math.OC", "stat.ML"
    "cs.CR", "cs.IT", "math.NT"
]

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
model = sentence_transformers.SentenceTransformer("all-MiniLM-L6-v2")

seen_ids = set()

total_inserted = 0

for category in CATEGORIES:
    print(f"\n📥 Fetching category: {category}")

    search = arxiv.Search(
        query=f"cat:{category}",
        max_results=MAX_RESULTS_PER_CATEGORY,
        sort_by=arxiv.SortCriterion.SubmittedDate
    )

    results = list(search.results())

    for paper in tqdm(results):
        arxiv_id = paper.entry_id.split("/")[-1]

        # Skip duplicates across categories or reruns
        if arxiv_id in seen_ids:
            continue
        seen_ids.add(arxiv_id)

        text = f"{paper.title}\n\n{paper.summary}"
        embedding = model.encode(text).tolist()

        data = {
            "arxiv_id": arxiv_id,
            "title": paper.title,
            "abstract": paper.summary,
            "authors": [author.name for author in paper.authors],
            "categories": paper.categories,
            "published_date": paper.published.date().isoformat(),
            "embedding": embedding
        }

        supabase.table("papers").upsert(
            data,
            on_conflict="arxiv_id"
        ).execute()

        total_inserted += 1

print(f"\n✅ Ingestion complete. Total unique papers processed: {total_inserted}")