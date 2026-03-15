import typer
import os
import json

from cli_second_brain.core import (
    delete_collection,
    embed_all_notes,
    get_backlinks,
    get_connected,
    get_neighbors_by_relative_path,
    get_links,
    search_by_filename_exact,
    search_notes,
    search_notes_graph,
)

app = typer.Typer(
    help="Second-brain CLI: manage and search your Obsidian vault with Qdrant embeddings",
    no_args_is_help=True
)

# ==========================
# EMBED
# ==========================
@app.command(help="Embed or update all notes in the vault")
def embed(
    force: bool = typer.Option(False, help="Re-embed all notes even if unchanged"),
    batch_size: int = typer.Option(None, help="Override batch size for embedding")
):
    """
    Embed or update all notes in your vault.

    Steps:
    1. Scan all Markdown notes in your vault.
    2. Compute embeddings using OpenAI and sparse embedding model.
    3. Upsert embeddings into the Qdrant collection.
    4. Skip notes that haven't changed unless --force is used.
    """
    final_batch_size = batch_size if batch_size is not None else int(os.getenv("SB_QDRANT_BATCH_SIZE", 64))
    updated = embed_all_notes(batch_size=final_batch_size, force_update=force)
    typer.echo(f"✅ Updated {updated} notes (batch size={final_batch_size})")


# ==========================
# SEARCH
# ==========================

def format_note_output(note: dict) -> dict:
    """
    Reorder CLI JSON output for readability:
    - filename, uuid, score
    - type, status, tags
    - content_summary
    - links
    - note_content (optional)
    """
    output = {
        "filename": note.get("filename"),
        "uuid": note.get("uuid"),
        "score": note.get("score"),
        "type": note.get("type"),
        "status": note.get("status"),
        "tags": note.get("tags"),
        "content_summary": note.get("content_summary"),
        "links": note.get("links"),
    }

    # Include full content at the end if present
    if "note_content" in note:
        output["note_content"] = note["note_content"]

    return output


@app.command(help="Search notes by semantic similarity with optional filters. Set --top-k 0 to return all results.")
def search(
    query: str = typer.Argument(None, help="Search query (leave empty to filter only)"),
    top_k: int | None = typer.Option(
        None,
        help="Maximum number of results (default from SB_DEFAULT_SEARCH_TOP_K, set 0 for all results)"
    ),
    min_score: float | None = typer.Option(
        None,
        help="Minimum similarity score (default from SB_DEFAULT_SEACRH_MIN_SCORE)"
    ),
    status: list[str] = typer.Option(None, help="Filter by note status (can provide multiple)"),
    folder: list[str] = typer.Option(None, help="Filter by folder(s) of the notes"),
    tags: list[str] = typer.Option(None, help="Filter by tags (can provide multiple)"),
    include_content: bool = typer.Option(False, help="Include the full note content in the results")
):
    matches = search_notes(
        query=query,
        top_k=top_k,
        min_score=min_score,
        status=status,
        folder=folder,
        tags=tags,
        include_content=include_content
    )
    formatted = [format_note_output(n) for n in matches]
    typer.echo(json.dumps(formatted, indent=2)) 


@app.command(name="search-graph", help="Graph-boosted semantic search with optional filters. Set --top-k 0 to return all results.")
def search_graph(
    query: str = typer.Argument(None, help="Search query (leave empty to filter only)"),
    top_k: int | None = typer.Option(
        None,
        help="Maximum number of results (default from SB_DEFAULT_SEARCH_TOP_K, set 0 for all results)"
    ),
    graph_boost: float = typer.Option(0.05, help="Score boost for graph neighbors"),
    status: list[str] = typer.Option(None, help="Filter by note status"),
    folder: list[str] = typer.Option(None, help="Filter by folder(s)"),
    tags: list[str] = typer.Option(None, help="Filter by tags"),
    min_score: float | None = typer.Option(
        None,
        help="Minimum similarity score (default from SB_DEFAULT_SEARCH_MIN_SCORE)"
    ),
    graph_expand: bool = typer.Option(False, help="Include neighbors not in initial results; they appear at the bottom"),
    include_content: bool = typer.Option(False, help="Include the full note content in the results")
):
    matches = search_notes_graph(
        query=query,
        top_k=top_k,
        graph_boost=graph_boost,
        graph_expand=graph_expand,
        min_score=min_score,
        include_content=include_content,
        status=status,
        folder=folder,
        tags=tags
    )
    formatted = [format_note_output(n) for n in matches]
    typer.echo(json.dumps(formatted, indent=2))


@app.command(help="Show graph neighbors (links + backlinks) of a note by relative path")
def neighbors(
    relative_path: str,
    status: list[str] = typer.Option(None, help="Filter by note status"),
    type_filter: list[str] = typer.Option(
        None,
        "--type",
        help="Filter by note type (can provide multiple)"
    ),
    include_content: bool = typer.Option(
        False,
        "--include-content",
        help="Include the full note content in the output"
    )
):
    """
    Retrieve graph neighbors for a note identified by relative path
    (folder/filename) with optional status/type filters and note content.
    """
    # Get neighbor metadata + content
    neighbors_data = get_neighbors_by_relative_path(
        relative_path,
        status=status,
        type_filter=type_filter,
        include_content=include_content
    )

    # Use the same output formatting as search
    formatted = [format_note_output(n) for n in neighbors_data]
    typer.echo(json.dumps(formatted, indent=2))


@app.command(help="Find notes by exact filename")
def filename(
    name: str,
    folder: list[str] = typer.Option(None, help="Restrict search to folders")
):
    matches = search_by_filename_exact(name, folders=folder)
    typer.echo(json.dumps(matches, indent=2))


# ==========================
# LINKS / BACKLINKS / GRAPH
# ==========================
@app.command(help="Show all outgoing links from a note")
def links(note: str, status: list[str] = None, folder: list[str] = None, tags: list[str] = None):
    typer.echo(json.dumps(get_links(note, status=status, folder=folder, tags=tags), indent=2))


@app.command(help="Show all backlinks to a note")
def backlinks(note: str, status: list[str] = None, folder: list[str] = None, tags: list[str] = None):
    typer.echo(json.dumps(get_backlinks(note, status=status, folder=folder, tags=tags), indent=2))


@app.command(help="Show both links and backlinks for a note")
def graph(note: str, status: list[str] = None, folder: list[str] = None, tags: list[str] = None):
    typer.echo(json.dumps(get_connected(note, status=status, folder=folder, tags=tags), indent=2))


# ==========================
# DELETE / REBUILD COLLECTION
# ==========================
@app.command(help="Fully rebuild the Qdrant index from scratch")
def rebuild(
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation prompt"),
    batch_size: int = typer.Option(None, help="Override batch size for embedding")
):
    """
    Rebuild the entire search index.

    Steps:
    1. Delete the Qdrant collection
    2. Clear the local index cache
    3. Re-embed all notes in the vault

    The embedding cache is preserved to avoid recomputing embeddings.
    """
    if not yes:
        confirm = typer.confirm(
            "This will delete the entire collection and rebuild the index. Continue?"
        )
        if not confirm:
            typer.echo("Aborted.")
            raise typer.Exit()

    typer.echo("🧹 Deleting collection...")
    delete_collection()

    final_batch_size = batch_size if batch_size is not None else int(
        os.getenv("SB_QDRANT_BATCH_SIZE", 1)
    )

    typer.echo("🔄 Re-embedding all notes...")
    updated = embed_all_notes(batch_size=final_batch_size, force_update=True)

    typer.echo(f"✅ Rebuild complete. Embedded {updated} notes.")


@app.command(name="delete-collection", help="Delete the entire Qdrant collection")
def delete_collection_cmd(
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation prompt")
):
    """
    Permanently delete the Qdrant collection.

    WARNING: This will remove all embeddings and metadata in the collection.
    Use --yes to skip the interactive confirmation.
    """
    if not yes:
        confirm = typer.confirm("This will permanently delete the entire collection. Continue?")
        if not confirm:
            typer.echo("Aborted.")
            raise typer.Exit()

    success = delete_collection()
    if success:
        typer.echo("Collection deleted.")
    else:
        typer.echo("Collection does not exist.")


if __name__ == "__main__":
    app()