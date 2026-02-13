"""CLI interface for the Journal Agent."""

import argparse
import json
import logging
import sys

from journal_agent.agent import JournalAgent


def main():
    parser = argparse.ArgumentParser(
        description="Journal Image Upload Agent - Upload, OCR, and search journal entries."
    )
    parser.add_argument(
        "--db", default="journal_entries.db", help="Path to the SQLite database file."
    )
    parser.add_argument(
        "--upload-dir", default="uploads", help="Directory for storing uploaded images."
    )
    parser.add_argument(
        "--ocr", choices=["tesseract", "google_vision"],
        default=None,
        help="OCR provider (default: env OCR_PROVIDER or tesseract).",
    )
    parser.add_argument(
        "--database", choices=["sqlite", "notion"],
        default=None,
        help="Database provider (default: env DATABASE_PROVIDER or sqlite).",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # --- ingest ---
    ingest_parser = subparsers.add_parser("ingest", help="Ingest a journal entry image.")
    ingest_parser.add_argument("image", help="Path to the image file.")
    ingest_parser.add_argument("--title", help="Title for the entry.")
    ingest_parser.add_argument("--tags", help="Comma-separated tags.")
    ingest_parser.add_argument("--date", help="Entry date (ISO format).")

    # --- search ---
    search_parser = subparsers.add_parser("search", help="Search journal entries.")
    search_parser.add_argument("query", help="Full-text search query.")
    search_parser.add_argument("--limit", type=int, default=20, help="Max results.")

    # --- list ---
    list_parser = subparsers.add_parser("list", help="List journal entries.")
    list_parser.add_argument("--tag", help="Filter by tag.")
    list_parser.add_argument("--limit", type=int, default=50, help="Max results.")

    # --- get ---
    get_parser = subparsers.add_parser("get", help="Get a journal entry by ID.")
    get_parser.add_argument("entry_id", help="Entry ID.")

    # --- delete ---
    delete_parser = subparsers.add_parser("delete", help="Delete a journal entry.")
    delete_parser.add_argument("entry_id", help="Entry ID.")

    # --- re-ocr ---
    reocr_parser = subparsers.add_parser("re-ocr", help="Re-run OCR on an existing entry.")
    reocr_parser.add_argument("entry_id", help="Entry ID.")

    # --- stats ---
    subparsers.add_parser("stats", help="Show database statistics.")

    # --- serve ---
    serve_parser = subparsers.add_parser("serve", help="Start the REST API server.")
    serve_parser.add_argument("--host", default="0.0.0.0", help="Bind host.")
    serve_parser.add_argument("--port", type=int, default=8000, help="Bind port.")

    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s"
    )

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    if args.command == "serve":
        import uvicorn
        uvicorn.run(
            "journal_agent.api:app",
            host=args.host,
            port=args.port,
            reload=False,
        )
        return

    def _parse_entry_id(raw: str):
        """Return int for SQLite IDs, string for Notion UUIDs."""
        try:
            return int(raw)
        except ValueError:
            return raw

    agent_kwargs = dict(
        db_path=args.db,
        upload_dir=args.upload_dir,
        ocr_provider=args.ocr,
        db_provider=args.database,
    )

    with JournalAgent(**agent_kwargs) as agent:
        if args.command == "ingest":
            tags = [t.strip() for t in args.tags.split(",") if t.strip()] if args.tags else None
            result = agent.ingest(
                image_path=args.image,
                title=args.title,
                tags=tags,
                entry_date=args.date,
            )
            print(json.dumps(result, indent=2))

        elif args.command == "search":
            results = agent.search(query=args.query, limit=args.limit)
            if not results:
                print("No results found.")
            else:
                print(f"Found {len(results)} result(s):\n")
                for r in results:
                    print(f"  [{r['id']}] {r.get('title') or '(untitled)'}")
                    preview = r["extracted_text"][:120].replace("\n", " ")
                    print(f"       {preview}...")
                    print()

        elif args.command == "list":
            entries = agent.list_entries(limit=args.limit, tag=args.tag)
            if not entries:
                print("No entries found.")
            else:
                for e in entries:
                    tags_str = f" [{e['tags']}]" if e["tags"] else ""
                    print(f"  [{e['id']}] {e.get('title') or '(untitled)'}{tags_str}  ({e['created_at'][:10]})")

        elif args.command == "get":
            entry = agent.get_entry(_parse_entry_id(args.entry_id))
            if not entry:
                print(f"Entry {args.entry_id} not found.")
                sys.exit(1)
            print(json.dumps(dict(entry), indent=2))

        elif args.command == "delete":
            if agent.delete_entry(_parse_entry_id(args.entry_id)):
                print(f"Deleted entry {args.entry_id}.")
            else:
                print(f"Entry {args.entry_id} not found.")
                sys.exit(1)

        elif args.command == "re-ocr":
            result = agent.re_ocr_entry(_parse_entry_id(args.entry_id))
            if not result:
                print(f"Entry {args.entry_id} not found.")
                sys.exit(1)
            print(f"Re-OCR complete. New text ({len(result['extracted_text'])} chars):")
            print(result["extracted_text"][:500])

        elif args.command == "stats":
            stats = agent.stats()
            print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
