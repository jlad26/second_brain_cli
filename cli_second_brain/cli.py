# second_brain_cli/cli.py
import typer
import os
import json
    
from cli_second_brain.core import embed_all_notes, search_notes

app = typer.Typer(help="Second-brain CLI using Qdrant and OpenAI embeddings")

@app.command()
def embed(
    force: bool = typer.Option(False, help="Re-embed all notes even if unchanged"),
    batch_size: int = typer.Option(None, help="Override batch size for embedding")
):
    """
    Embed / update all notes in the second brain.
    """
    final_batch_size = batch_size if batch_size is not None else int(os.getenv("SB_QDRANT_BATCH_SIZE", 1))
    updated = embed_all_notes(batch_size=final_batch_size, force_update=force)
    typer.echo(f"✅ Updated {updated} notes (batch size={final_batch_size})")

@app.command()
def search(
    query: str,
    top_k: int = typer.Option(5, help="Number of results to return"),
    type_filter: str = typer.Option(None, help="Filter results by note type"),
    min_score: float = typer.Option(None, help="Minimum similarity score (0–1) threshold"),
    json_output: bool = typer.Option(True, help="Output results as JSON")
):
    """
    Search notes by query.

    Options:
    --top-k : Maximum number of results
    --type-filter : Filter by note type (e.g., project, idea)
    --min-score : Minimum similarity score threshold
    --json-output : Output results as JSON string
    """
    matches = search_notes(
        query=query,
        top_k=top_k,
        type_filter=type_filter,
        min_score=min_score
    )

    if not matches:
        if json_output:
            typer.echo(json.dumps([]))  # return empty JSON array
        else:
            typer.echo("No matches found.")
        return

    if json_output:
        # Convert all matches to JSON string
        typer.echo(json.dumps(matches, indent=2))
    else:
        # Fallback: human-readable format
        for m in matches:
            typer.echo(
                f"{m['filename']} - score: {m['score']:.3f} - type: {m['type']} - "
                f"tags: {m['tags']} - status: {m['status']}"
            )

if __name__ == "__main__":
    app()