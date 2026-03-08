import hashlib
import json
import os
import uuid
import re
from collections import defaultdict
from dotenv import load_dotenv
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
    PointIdsList,
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

def load_env():
    """Load environment variables for the CLI."""
    
    # User config directory
    user_env = Path.home() / ".config" / "second-brain" / ".env"
    
    # Local project .env
    local_env = Path.cwd() / ".env"

    if user_env.exists():
        load_dotenv(user_env)

    if local_env.exists():
        load_dotenv(local_env, override=True)

load_env()

def get_cache_dir():
    env_dir = os.getenv("SB_CACHE_DIR")

    if env_dir:
        path = Path(env_dir).expanduser()
    else:
        path = Path.home() / ".cache" / "second-brain"

    path.mkdir(parents=True, exist_ok=True)
    return path


CACHE_DIR = get_cache_dir()

SB_QDRANT_HF_TOKEN = os.getenv("SB_QDRANT_HF_TOKEN", "")
QDRANT_URL = os.getenv("SB_QDRANT_URL")
QDRANT_API_KEY = os.getenv("SB_QDRANT_API_KEY")
COLLECTION_NAME = os.getenv("SB_QDRANT_COLLECTION_NAME", "")
OPENAI_API_KEY = os.getenv("SB_QDRANT_OPENAI_API_KEY")
NOTES_DIR = Path(os.getenv("SB_QDRANT_NOTES_DIR", "")).expanduser().resolve()
if not NOTES_DIR.exists():
    raise ValueError(f"NOTES_DIR does not exist: {NOTES_DIR}")
EMBED_MODEL = os.getenv("SB_QDRANT_EMBED_MODEL", "")
BATCH_SIZE = int(os.getenv("SB_QDRANT_BATCH_SIZE", 64))
GRAPH_NEIGHBOR_LIMIT = int(os.getenv("SB_GRAPH_NEIGHBOR_LIMIT", 20))

CACHE_FILE = os.getenv(
    "SB_QDRANT_CACHE_FILE",
    str(CACHE_DIR / "embedding_cache.json")
)

INDEX_CACHE_FILE = os.getenv(
    "SB_QDRANT_INDEX_CACHE",
    str(CACHE_DIR / "index_cache.json")
)

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

    qdrant.create_payload_index(COLLECTION_NAME, "status", PayloadSchemaType.KEYWORD)
    qdrant.create_payload_index(COLLECTION_NAME, "type", PayloadSchemaType.KEYWORD)
    qdrant.create_payload_index(COLLECTION_NAME, "tags", PayloadSchemaType.KEYWORD)
    qdrant.create_payload_index(COLLECTION_NAME, "folder", PayloadSchemaType.KEYWORD)
    qdrant.create_payload_index(COLLECTION_NAME, "links", PayloadSchemaType.KEYWORD)
    qdrant.create_payload_index(COLLECTION_NAME, "filename", PayloadSchemaType.KEYWORD)
    qdrant.create_payload_index(COLLECTION_NAME, "uuid", PayloadSchemaType.KEYWORD)

# ==========================
# HELPERS
# ==========================

LINK_PATTERN = r"\[\[([^\]]+)\]\]"


def extract_links(text: str):

    matches = re.findall(LINK_PATTERN, text)

    cleaned = []

    for m in matches:
        m = m.split("|")[0]
        m = m.split("/")[-1]
        cleaned.append(m.strip())

    return cleaned


def file_hash(text: str):
    payload = f"{EMBED_MODEL}:{text}"
    return hashlib.sha256(payload.encode()).hexdigest()


def note_uuid(path: Path):
    return str(uuid.uuid5(uuid.NAMESPACE_URL, str(path)))


def resolve_uuids_to_filenames(uuids: list[str]):
    if not uuids:
        return []

    points = qdrant.retrieve(
        collection_name=COLLECTION_NAME,
        ids=uuids
    )

    uuid_to_filename = {
        r.payload["uuid"]: r.payload["filename"]
        for r in points
        if r.payload
    }

    return [uuid_to_filename.get(u, u) for u in uuids]

# ==========================
# EMBEDDING CACHE
# ==========================

def atomic_write_json(path: Path, data):
    tmp_path = path.with_suffix(".tmp")

    with open(tmp_path, "w") as f:
        json.dump(data, f)

    tmp_path.replace(path)


def load_cache():
    path = Path(CACHE_FILE)

    if not path.exists():
        return {}

    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        print("⚠ embedding cache corrupted")
        return {}


def save_cache(cache):
    atomic_write_json(Path(CACHE_FILE), cache)


def load_index_cache():
    path = Path(INDEX_CACHE_FILE)

    if not path.exists():
        return {}

    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        print("⚠ index cache corrupted")
        return {}


def save_index_cache(cache):
    atomic_write_json(Path(INDEX_CACHE_FILE), cache)


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

    return embeddings


# ==========================
# EMBEDDING PIPELINE
# ==========================

def scan_notes(vault_dir: Path):
    """
    Recursively scan vault_dir for Markdown files, skipping hidden files/folders.
    Returns a list of Path objects.
    """
    note_paths = []
    vault_dir = vault_dir.resolve()

    for root, dirs, files in os.walk(vault_dir):
        # Skip hidden directories
        dirs[:] = [d for d in dirs if not d.startswith(".")]

        for f in files:
            if f.startswith(".") or not f.endswith(".md"):
                continue
            note_paths.append(Path(root) / f)

    return note_paths


def embed_all_notes(batch_size=BATCH_SIZE, force_update=False):
    """
    Scan the NOTES_DIR vault, compute embeddings for new or changed notes,
    and upsert them into Qdrant, updating embedding and index caches.
    """
    NOTES_DIR_path = Path(NOTES_DIR).expanduser().resolve()
    if not NOTES_DIR_path.exists():
        raise ValueError(f"NOTES_DIR does not exist: {NOTES_DIR_path}")

    # ---------------------------------------
    # Load existing UUID -> hash from Qdrant
    # ---------------------------------------
    existing_hashes = load_index_cache()
    
    if COLLECTION_NAME not in [c.name for c in qdrant.get_collections().collections]:
        print("⚠ Qdrant collection missing, clearing local index cache")
        existing_hashes = {}

    # ---------------------------------------
    # Scan vault (fast, relative hidden filtering)
    # ---------------------------------------
    note_paths = scan_notes(NOTES_DIR_path)
    print(f"Vault scan found {len(note_paths)} markdown files")

    # Build filename -> path index
    filename_index = defaultdict(list)
    for p in note_paths:
        filename_index[p.stem].append(p)

    notes_to_embed = []

    for note_path in note_paths:
        note = frontmatter.load(str(note_path))
        text = note.content.strip()
        if not text:
            continue

        note_id = note_uuid(note_path)

        # Resolve links
        link_names = extract_links(text)
        link_uuids = []
        for lname in link_names:
            for candidate in filename_index.get(lname, []):
                link_uuids.append(note_uuid(candidate))

        relative_path = note_path.relative_to(NOTES_DIR_path)

        status = note.get("status", "active")
        note_type = note.get("type", "idea")
        tags = note.get("tags", [])
        if isinstance(tags, str):
            tags = [tags]

        # Compute hash including metadata
        hash_source = json.dumps(
            {
                "content": text,
                "status": status,
                "type": note_type,
                "tags": tags,
                "links": sorted(link_uuids),
                "filename": str(relative_path),
            },
            sort_keys=True,
        )

        current_hash = file_hash(hash_source)

        if not force_update and existing_hashes.get(note_id) == current_hash:
            continue

        notes_to_embed.append({
            "id": note_id,
            "text": text,
            "payload": {
                "filename": str(relative_path),
                "uuid": note_id,
                "folder": str(relative_path.parent),
                "status": status,
                "type": note_type,
                "tags": tags,
                "links": link_uuids,
                "hash": current_hash
            }
        })

    updated = 0

    # Detect deleted notes
    current_note_ids = {note_uuid(p) for p in note_paths}
    cached_note_ids = set(existing_hashes.keys())
    deleted_ids = cached_note_ids - current_note_ids

    if deleted_ids:
        print(f"Removing {len(deleted_ids)} deleted notes from index")
        qdrant.delete(
            collection_name=COLLECTION_NAME,
            points_selector=PointIdsList(points=list(deleted_ids))
        )
        for did in deleted_ids:
            existing_hashes.pop(did, None)

    print(f"{len(notes_to_embed)} notes require embedding")

    # ---------------------------------------
    # Embed + upsert in batches
    # ---------------------------------------
    for i in range(0, len(notes_to_embed), batch_size):
        batch = notes_to_embed[i:i + batch_size]
        texts = [n["text"] for n in batch]

        dense_vectors = get_embeddings(texts)
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

        qdrant.upsert(collection_name=COLLECTION_NAME, points=points)

        # Update index cache
        for n in batch:
            existing_hashes[n["id"]] = n["payload"]["hash"]

        updated += len(points)
        print(f"Embedded {updated}/{len(notes_to_embed)}")

    # Save caches
    save_cache(embedding_cache)
    save_index_cache(existing_hashes)

    return updated


# ==========================
# FILTER BUILDER
# ==========================

def build_filter(type_filter=None, status=None, folder=None, tags=None):
    conditions = []

    if status:
        conditions.append(FieldCondition(key="status", match=MatchValue(value=status[0])) if len(status) == 1 else None)
    if type_filter:
        conditions.append(FieldCondition(key="type", match=MatchValue(value=type_filter)))
    if folder:
        conditions.append(FieldCondition(key="folder", match=MatchValue(value=folder[0])) if len(folder) == 1 else None)
    if tags:
        conditions.append(FieldCondition(key="tags", match=MatchValue(value=tags[0])) if len(tags) == 1 else None)

    conditions = [c for c in conditions if c is not None]
    if not conditions:
        return None
    return Filter(must=conditions)


# ==========================
# SEARCH
# ==========================

def search_notes(query=None, top_k=5, min_score=None, include_content=False, **filters):
    """
    Search notes using semantic similarity and optional filters.

    Args:
        query (str | None): Semantic search query. If None, performs a filter-only search.
        top_k (int): Maximum number of results. Set 0 to return all matching notes.
        min_score (float | None): Minimum similarity score threshold to filter results.
        include_content (bool): Whether to include the full note content in the output.
        **filters: Optional filters including status, folder, tags, type, etc.

    Returns:
        list[dict]: A list of note entries with metadata and optional content.

    Notes:
        - If query is None, only filtering is applied (no semantic search).
        - If top_k=0, all notes matching the filters (and query if given) are returned.
    """
    payload_filter = build_filter(**filters)

    prefetch = []
    fusion_query = None

    if query:
        query_emb = openai_client.embeddings.create(
            input=query,
            model=EMBED_MODEL
        ).data[0].embedding

        sparse_query = list(sparse_model.embed([query]))[0]
        sparse_vector = SparseVector(
            indices=list(sparse_query.indices),
            values=list(sparse_query.values)
        )

        prefetch = [
            Prefetch(query=query_emb, using=""),
            Prefetch(query=sparse_vector, using="text")
        ]
        fusion_query = FusionQuery(fusion=Fusion.RRF)

    # If top_k=0, fetch all matching points
    qdrant_limit = top_k if top_k > 0 else 1000000  # arbitrarily large for "all"

    results = qdrant.query_points(
        collection_name=COLLECTION_NAME,
        prefetch=prefetch if prefetch else None,
        query=fusion_query,
        limit=qdrant_limit * 2 if top_k > 0 else qdrant_limit,  # maintain previous *2 only if top_k>0
        query_filter=payload_filter
    )

    hits = results.points
    if min_score:
        hits = [h for h in hits if h.score >= min_score]

    # Slice to top_k only if top_k>0
    hits = hits if top_k == 0 else hits[:top_k]

    output = []
    for h in hits:
        if not h.payload:
            continue

        entry = {
            "filename": h.payload["filename"],
            "uuid": h.payload.get("uuid"),
            "score": h.score,
            "type": h.payload.get("type"),
            "status": h.payload.get("status"),
            "tags": h.payload.get("tags"),
            "links": h.payload.get("links")
        }

        if include_content:
            file_path = Path(NOTES_DIR) / h.payload["filename"]
            if file_path.exists():
                entry["note_content"] = file_path.read_text(encoding="utf-8")
            else:
                entry["note_content"] = None

        output.append(entry)

    return output


# ==========================
# GRAPH
# ==========================

def get_links(note_uuid, status=None, folder=None, tags=None, limit=100):
    payload_filter = build_filter(status=status, folder=folder, tags=tags)
    payload_filter = Filter(
        must=[FieldCondition(key="uuid", match=MatchValue(value=note_uuid))],
        should=[payload_filter] if payload_filter else None
    )
    points, _ = qdrant.scroll(collection_name=COLLECTION_NAME, scroll_filter=payload_filter, limit=limit)
    links = set()
    for r in points:
        if r.payload and "links" in r.payload:
            links.update(r.payload["links"])  # now UUIDs
    return list(links)

def get_backlinks(note_uuid, status=None, folder=None, tags=None, limit=100):
    payload_filter = build_filter(status=status, folder=folder, tags=tags)
    payload_filter = Filter(
        must=[FieldCondition(key="links", match=MatchValue(value=note_uuid))],
        should=[payload_filter] if payload_filter else None
    )
    points, _ = qdrant.scroll(collection_name=COLLECTION_NAME, scroll_filter=payload_filter, limit=limit)
    return [r.payload["uuid"] for r in points if r.payload]

def get_connected(note_name, status=None, folder=None, tags=None):
    return {
        "outgoing": get_links(note_name, status=status, folder=folder, tags=tags),
        "incoming": get_backlinks(note_name, status=status, folder=folder, tags=tags)
    }

def get_graph_neighbors(note_name, status=None, folder=None, tags=None, limit=None):
    """
    Returns all graph neighbors (links + backlinks) respecting filters.
    """
    if limit is None:
        limit = GRAPH_NEIGHBOR_LIMIT

    neighbors = set()
    neighbors.update(get_links(note_name, status=status, folder=folder, tags=tags, limit=limit))
    neighbors.update(get_backlinks(note_name, status=status, folder=folder, tags=tags, limit=limit))
    return list(neighbors)

def get_neighbors_by_relative_path(relative_path: str, status=None, type_filter: list[str] | None = None):
    """
    Return graph neighbors (links + backlinks) for a note identified
    by its relative path (folder/filename). Returns filenames instead of UUIDs.

    Args:
        relative_path: string like "Interests/Artificial Intelligence"
        status: optional list of note statuses to filter
        type_filter: optional list of note types to filter neighbors (e.g., ["idea", "project"])
    """
    from pathlib import Path

    # Ensure relative path has .md extension
    path_obj = Path(relative_path)
    if path_obj.suffix != ".md":
        path_obj = path_obj.with_suffix(".md")

    relative_path_str = str(path_obj)

    # Lookup the note in Qdrant
    points, _ = qdrant.scroll(
        collection_name=COLLECTION_NAME,
        scroll_filter=Filter(
            must=[FieldCondition(key="filename", match=MatchValue(value=relative_path_str))]
        ),
        limit=1
    )

    if not points or not points[0].payload:
        return []

    note_uuid = points[0].payload["uuid"]

    # Fetch neighbors with optional status filter
    neighbors = get_graph_neighbors(note_uuid, status=status, folder=None, tags=None, limit=None)

    # Filter neighbors by type(s) if provided
    if type_filter:
        filtered_neighbors = []
        points = qdrant.retrieve(collection_name=COLLECTION_NAME, ids=neighbors)
        for p in points:
            if p.payload and p.payload.get("type") in type_filter:
                filtered_neighbors.append(p.payload["uuid"])
        neighbors = filtered_neighbors

    # Convert UUIDs → filenames
    return resolve_uuids_to_filenames(neighbors)

def graph_rerank(
    results, 
    boost=0.05, 
    expand=False, 
    status=None, 
    folder=None, 
    tags=None
):
    # Use UUID as key
    scores = {r["uuid"]: r["score"] for r in results}
    payloads = {r["uuid"]: r for r in results}

    for r in results:
        note_id = r["uuid"]
        neighbors = get_graph_neighbors(note_id, status=status, folder=folder, tags=tags)

        for n in neighbors:
            if n in scores:
                # boost existing neighbor
                scores[n] += boost
            elif expand:
                # fetch neighbor by UUID
                existing = qdrant.retrieve(collection_name=COLLECTION_NAME, ids=[n])
                if existing and existing[0].payload:
                    scores[n] = 0
                    payloads[n] = {
                        "filename": existing[0].payload.get("filename", ""),
                        "score": 0,
                        "uuid": n,
                        "type": existing[0].payload.get("type"),
                        "status": existing[0].payload.get("status"),
                        "tags": existing[0].payload.get("tags", []),
                        "links": existing[0].payload.get("links", [])
                    }

    # Reconstruct sorted list
    reranked = [
        payloads[uid] for uid, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)
    ]
    return reranked


def search_notes_graph(
    query=None,
    top_k=5,
    graph_boost=0.05,
    graph_expand=False,
    min_score=None,
    include_content=False,
    **filters
):
    """
    Perform a graph-boosted semantic search with optional filters.

    Args:
        query (str | None): Semantic search query. If None, performs filter-only search.
        top_k (int): Maximum number of results. Set 0 to return all matching notes.
        graph_boost (float): Score boost applied to graph neighbors.
        graph_expand (bool): If True, includes neighbors not in initial results.
        min_score (float | None): Minimum similarity score for the initial semantic search.
        include_content (bool): Whether to include the full note content in the output.
        **filters: Optional filters including status, folder, tags, type, etc.

    Returns:
        list[dict]: A list of notes with reranked scores, metadata, and optional content.

    Notes:
        - If query is None, only filters are applied (no semantic search).
        - If top_k=0, all notes matching the filters (and query if given) are returned.
        - Graph boosting applies to existing search results and optionally to neighbors
          if `graph_expand` is True.
    """
    # Determine how many initial results to fetch
    initial_top_k = top_k * 3 if top_k > 0 else 1000000  # fetch all if top_k=0

    # Initial search (semantic + filters)
    results = search_notes(
        query=query,
        top_k=initial_top_k,
        min_score=min_score,
        include_content=include_content,
        **filters
    )

    # Rerank and optionally expand neighbors
    reranked = graph_rerank(
        results,
        boost=graph_boost,
        expand=graph_expand,
        **filters
    )

    # Fetch file content for any neighbors added via expand
    if include_content:
        for r in reranked:
            if "note_content" not in r:
                file_path = Path(NOTES_DIR) / r["filename"]
                if file_path.exists():
                    r["note_content"] = file_path.read_text(encoding="utf-8")
                else:
                    r["note_content"] = None

    # Convert neighbor links UUIDs to filenames for CLI
    for r in reranked:
        if "links" in r:
            r["links"] = resolve_uuids_to_filenames(r["links"])

    # Slice final results only if top_k > 0
    return reranked if top_k == 0 else reranked[:top_k]


from pathlib import Path

def search_by_filename_exact(
    filename: str,
    folders: list[str] | None = None
):
    """
    Return notes whose filename exactly matches the given name,
    excluding hidden files/folders (starting with a dot).

    Args:
        filename: exact note name (without .md)
        folders: optional list of folders to restrict search

    Returns:
        list of matches
    """

    results = []

    for path in Path(NOTES_DIR).rglob("*.md"):

        # Exclude hidden files
        if path.name.startswith("."):
            continue

        # Exclude any path that has a folder starting with a dot
        if any(part.startswith(".") for part in path.parts):
            continue

        relative = path.relative_to(NOTES_DIR)

        # Optional folder filter
        if folders:
            if not any(str(relative.parent).startswith(f) for f in folders):
                continue

        # Exact filename match (case-insensitive)
        if path.stem.lower() == filename.lower():
            results.append(str(relative))

    return results


def delete_collection():

    collections = [c.name for c in qdrant.get_collections().collections]

    if COLLECTION_NAME not in collections:
        return False

    qdrant.delete_collection(collection_name=COLLECTION_NAME)
    
    # Clear index cache
    index_path = Path(INDEX_CACHE_FILE)
    if index_path.exists():
        index_path.unlink()

    return True