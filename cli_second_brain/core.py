import hashlib
import json
import os
import uuid
import re
from pathlib import Path
from typing import Optional

import frontmatter
from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    Distance,
    Filter,
    FieldCondition,
    Fusion,
    FusionQuery,
    MatchValue,
    PayloadSchemaType,
    Prefetch,
    SparseVector,
    SparseVectorParams,
    VectorParams
)

from openai import OpenAI
from fastembed import SparseTextEmbedding

# ==========================
# CONFIGURATION
# ==========================

current_file_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_file_dir)
env_file_path = os.path.join(parent_dir, ".env")
if os.path.exists(env_file_path):
    from dotenv import load_dotenv
    load_dotenv()

SB_QDRANT_HF_TOKEN = os.getenv("SB_QDRANT_HF_TOKEN", "")
QDRANT_URL = os.getenv("SB_QDRANT_URL")
QDRANT_API_KEY = os.getenv("SB_QDRANT_API_KEY")
COLLECTION_NAME = os.getenv("SB_QDRANT_COLLECTION_NAME", "")
CACHE_FILE = os.getenv("SB_QDRANT_CACHE_FILE", "")
OPENAI_API_KEY = os.getenv("SB_QDRANT_OPENAI_API_KEY")
NOTES_DIR = os.getenv("SB_QDRANT_NOTES_DIR", "")
EMBED_MODEL = os.getenv("SB_QDRANT_EMBED_MODEL", "")
BATCH_SIZE = int(os.getenv("SB_QDRANT_BATCH_SIZE", 1))

os.environ["HF_TOKEN"] = SB_QDRANT_HF_TOKEN

# ==========================
# CLIENTS
# ==========================

qdrant = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY
)

openai_client = OpenAI(api_key=OPENAI_API_KEY)

sparse_model = SparseTextEmbedding(model_name="Qdrant/bm25")


# ==========================
# COLLECTION SETUP
# ==========================

collections = [c.name for c in qdrant.get_collections().collections]

if COLLECTION_NAME not in collections:

    qdrant.create_collection(
        collection_name=COLLECTION_NAME,

        vectors_config=VectorParams(
            size=1536,
            distance=Distance.COSINE
        ),

        sparse_vectors_config={
            "text": SparseVectorParams()
        }
    )
    
    # metadata indexes
    qdrant.create_payload_index(
        COLLECTION_NAME,
        "status",
        PayloadSchemaType.KEYWORD
    )

    qdrant.create_payload_index(
        COLLECTION_NAME,
        "type",
        PayloadSchemaType.KEYWORD
    )

    qdrant.create_payload_index(
        COLLECTION_NAME,
        "tags",
        PayloadSchemaType.KEYWORD
    )


# ==========================
# HELPERS
# ==========================

LINK_PATTERN = r"\[\[([^\]|]+)"


def extract_links(text: str):
    return re.findall(LINK_PATTERN, text)


def file_hash(text: str):
    return hashlib.sha256(text.encode()).hexdigest()


def note_uuid(path: Path):
    """Deterministic UUID based on file path"""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, str(path)))

# ==========================
# EMBEDDING CACHING
# ==========================

def load_cache():
    if Path(CACHE_FILE).exists():
        with open(CACHE_FILE, "r") as f:
            return json.load(f)
    return {}


def save_cache(cache):
    with open(CACHE_FILE, "w") as f:
        json.dump(cache, f)

embedding_cache = load_cache()

def get_embeddings(texts):

    new_texts = []
    new_indices = []

    embeddings: list[Optional[list[float]]] = [None] * len(texts)

    for i, text in enumerate(texts):

        h = file_hash(text)

        if h in embedding_cache:

            embeddings[i] = embedding_cache[h]

        else:

            new_texts.append(text)
            new_indices.append(i)

    if new_texts:

        response = openai_client.embeddings.create(
            input=new_texts,
            model=EMBED_MODEL
        )

        for idx, emb in zip(new_indices, response.data):

            vector = emb.embedding

            embeddings[idx] = vector

            embedding_cache[file_hash(texts[idx])] = vector

    save_cache(embedding_cache)

    return embeddings

# ==========================
# EMBEDDING PIPELINE
# ==========================

def embed_all_notes(batch_size=BATCH_SIZE, force_update=False):

    note_paths = list(Path(NOTES_DIR).rglob("*.md"))

    notes_to_embed = []

    for note_path in note_paths:

        note = frontmatter.load(str(note_path))

        text = note.content.strip()

        if not text:
            continue

        note_id = note_uuid(note_path)

        current_hash = file_hash(text)

        existing = qdrant.retrieve(
            collection_name=COLLECTION_NAME,
            ids=[note_id]
        )

        if existing and not force_update:

            payload = existing[0].payload
            if payload is not None and payload.get("hash") == current_hash:
                continue

        links = extract_links(text)

        notes_to_embed.append({

            "id": note_id,

            "text": text,

            "payload": {
                "filename": str(note_path),
                "status": note.get("status", "active"),
                "type": note.get("type", "idea"),
                "tags": note.get("tags", []),
                "links": links,
                "hash": current_hash
            }
        })

    print(f"Found {len(notes_to_embed)} notes to embed/update")

    updated = 0

    for i in range(0, len(notes_to_embed), batch_size):

        batch = notes_to_embed[i:i + batch_size]

        texts = [n["text"] for n in batch]

        # Dense embeddings
        dense_vectors = get_embeddings(texts)

        # Sparse embeddings
        sparse_vectors = list(sparse_model.embed(texts))

        points = []

        for n, dense, sparse in zip(batch, dense_vectors, sparse_vectors):

            points.append({
                "id": n["id"],
                "vector": {
                    "": dense,
                    "text": sparse.as_object()
                },
                "payload": n["payload"]
            })

        qdrant.upsert(
            collection_name=COLLECTION_NAME,
            points=points
        )

        updated += len(points)

        print(f"Embedded batch {i} → {i + len(batch) - 1}")

    return updated


# ==========================
# FILTER BUILDER
# ==========================

def build_filter(include_archived=False, type_filter=None):

    conditions = []

    if not include_archived:

        conditions.append(
            FieldCondition(
                key="status",
                match=MatchValue(value="active")
            )
        )

    if type_filter:

        conditions.append(
            FieldCondition(
                key="type",
                match=MatchValue(value=type_filter)
            )
        )

    if not conditions:
        return None

    return Filter(must=conditions)


# ==========================
# HYBRID SEARCH
# ==========================

def search_notes(
    query,
    top_k=5,
    include_archived=False,
    type_filter=None,
    min_score=None
):

    query_emb = openai_client.embeddings.create(
        input=query,
        model=EMBED_MODEL
    ).data[0].embedding

    sparse_query = list(sparse_model.embed([query]))[0]
    
    sparse_vector = SparseVector(
        indices=list(sparse_query.indices),
        values=list(sparse_query.values)
    )

    payload_filter = build_filter(
        include_archived=include_archived,
        type_filter=type_filter
    )

    results = qdrant.query_points(

        collection_name=COLLECTION_NAME,

        prefetch=[
            Prefetch(
                query=query_emb,
                using=""
            ),
            Prefetch(
                query=sparse_vector,
                using="text"
            )
        ],
        query=FusionQuery(
            fusion=Fusion.RRF
        ),
        limit=top_k * 2,
        query_filter=payload_filter
    )

    hits = results.points

    if min_score:
        hits = [h for h in hits if h.score >= min_score]

    hits = hits[:top_k]

    return [
        {
            "filename": h.payload["filename"],
            "score": h.score,
            "type": h.payload.get("type"),
            "status": h.payload.get("status"),
            "tags": h.payload.get("tags"),
            "links": h.payload.get("links")
        }
        for h in hits
        if h.payload is not None
    ]