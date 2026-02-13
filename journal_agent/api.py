"""FastAPI server exposing the Journal Agent as a REST API."""

import logging
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel

from journal_agent.agent import JournalAgent

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

agent: Optional[JournalAgent] = None


def get_agent() -> JournalAgent:
    global agent
    if agent is None:
        agent = JournalAgent()
        agent.start()
    return agent


@asynccontextmanager
async def lifespan(app: FastAPI):
    get_agent()
    logger.info("Journal Agent API started.")
    yield
    global agent
    if agent:
        agent.stop()
        agent = None


app = FastAPI(
    title="Journal Image Upload Agent",
    description="Upload images of journal entries, extract text via OCR, and search across all entries.",
    version="1.0.0",
    lifespan=lifespan,
)


# --- Request/Response models ---


class IngestResponse(BaseModel):
    id: int
    image_path: str
    extracted_text: str
    title: Optional[str] = None
    tags: list[str] = []
    entry_date: Optional[str] = None


class SearchResult(BaseModel):
    id: int
    image_path: str
    extracted_text: str
    title: Optional[str] = None
    tags: str = ""
    entry_date: Optional[str] = None
    created_at: str
    rank: float


class EntryResponse(BaseModel):
    id: int
    image_path: str
    extracted_text: str
    title: Optional[str] = None
    tags: str = ""
    entry_date: Optional[str] = None
    created_at: str
    updated_at: str


class UpdateRequest(BaseModel):
    title: Optional[str] = None
    tags: Optional[list[str]] = None


class StatsResponse(BaseModel):
    total_entries: int
    upload_dir: str
    db_path: str


# --- Endpoints ---


@app.post("/entries/upload", response_model=IngestResponse)
async def upload_journal_entry(
    image: UploadFile = File(...),
    title: Optional[str] = Query(None, description="Title for the journal entry"),
    tags: Optional[str] = Query(None, description="Comma-separated tags"),
    entry_date: Optional[str] = Query(None, description="Date of the entry (ISO format)"),
):
    """Upload a journal entry image. The image is processed with OCR and stored for search."""
    allowed_types = {"image/png", "image/jpeg", "image/tiff", "image/bmp", "image/webp"}
    if image.content_type and image.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{image.content_type}'. Allowed: {', '.join(sorted(allowed_types))}",
        )

    # Save uploaded file to a temp location, then ingest
    suffix = Path(image.filename).suffix if image.filename else ".png"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        contents = await image.read()
        tmp.write(contents)
        tmp_path = tmp.name

    tag_list = [t.strip() for t in tags.split(",") if t.strip()] if tags else None

    try:
        result = get_agent().ingest(
            image_path=tmp_path,
            title=title,
            tags=tag_list,
            entry_date=entry_date,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    return IngestResponse(**result)


@app.get("/entries/search", response_model=list[SearchResult])
def search_entries(
    q: str = Query(..., description="Full-text search query"),
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
):
    """Search journal entries by text content. Supports FTS5 syntax (AND, OR, NEAR, etc.)."""
    results = get_agent().search(query=q, limit=limit, offset=offset)
    return results


@app.get("/entries", response_model=list[EntryResponse])
def list_entries(
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    tag: Optional[str] = Query(None, description="Filter by tag"),
):
    """List all journal entries, optionally filtered by tag."""
    return get_agent().list_entries(limit=limit, offset=offset, tag=tag)


@app.get("/entries/{entry_id}", response_model=EntryResponse)
def get_entry(entry_id: int):
    """Get a single journal entry by ID."""
    entry = get_agent().get_entry(entry_id)
    if not entry:
        raise HTTPException(status_code=404, detail="Entry not found")
    return entry


@app.get("/entries/{entry_id}/image")
def get_entry_image(entry_id: int):
    """Download the original image for a journal entry."""
    entry = get_agent().get_entry(entry_id)
    if not entry:
        raise HTTPException(status_code=404, detail="Entry not found")

    image_path = Path(entry["image_path"])
    if not image_path.exists():
        raise HTTPException(status_code=404, detail="Image file not found on disk")

    return FileResponse(image_path)


@app.patch("/entries/{entry_id}")
def update_entry(entry_id: int, body: UpdateRequest):
    """Update the title or tags on a journal entry."""
    updated = get_agent().update_entry(
        entry_id, title=body.title, tags=body.tags
    )
    if not updated:
        raise HTTPException(status_code=404, detail="Entry not found")
    return get_agent().get_entry(entry_id)


@app.delete("/entries/{entry_id}")
def delete_entry(entry_id: int):
    """Delete a journal entry and its image."""
    deleted = get_agent().delete_entry(entry_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Entry not found")
    return {"deleted": True, "id": entry_id}


@app.post("/entries/{entry_id}/re-ocr")
def re_ocr_entry(entry_id: int):
    """Re-run OCR on an existing entry's image (useful after OCR improvements)."""
    result = get_agent().re_ocr_entry(entry_id)
    if not result:
        raise HTTPException(status_code=404, detail="Entry not found")
    return result


@app.get("/stats", response_model=StatsResponse)
def get_stats():
    """Get summary statistics about the journal database."""
    return get_agent().stats()
