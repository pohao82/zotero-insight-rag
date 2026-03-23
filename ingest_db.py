import os
import re
import argparse
from pathlib import Path
from app.utils.zotero import ZoteroLocalClient
from app.ingestion.parser import DocumentParser
from app.ingestion.database_schema import VectorDatabase
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# --- Configuration ---
# Models
MODE = "marker"  # Options: "pypdf", "marker"
USE_GPU = True

# File and folder Paths
#ZOTERO_STORAGE = "/workspace/data/Zotero/storage" # if use container
#DB_DIR = Path("/workspace/data/database")         # if use container
ZOTERO_STORAGE = Path("/home/phchang/Zotero/storage")
DB_DIR = Path("/home/phchang/db_zotero_rag")
BASE_NAME = "zotero_lieb_1650_200_550_120_m8_ef120_test"

# Stored parsed texts for reuse (marker is expensive) 
PARSED_CACHE = DB_DIR / f"{MODE}_cache_test"
DB_PATH = DB_DIR / f"{BASE_NAME}_{MODE}.db" 

EMBED_MODEL = "mxbai-embed-large"

# Parameters for chunking
PARENT_CHUNK_SIZE = 1650
PARENT_OVERLAP = 200
CHILD_CHUNK_SIZE = 550
CHILD_OVERLAP = 120

# --- User defined section ends ---

# Splitters
parent_splitter = RecursiveCharacterTextSplitter(
    chunk_size=PARENT_CHUNK_SIZE, 
    chunk_overlap=PARENT_OVERLAP,
    separators=["\n\n", "\n"]
)
child_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHILD_CHUNK_SIZE, 
    chunk_overlap=CHILD_OVERLAP,
    separators=["\n", ". ", " ", ""]
)

embeddings_model = OllamaEmbeddings(model=EMBED_MODEL)

# --- Helper Functions ---

def clean_scientific_text(text):
    """Refines stoichiometry by removing HTML sub/sup tags."""
    text = re.sub(r'\s*<sub>\s*(\d+)\s*</sub>\s*', r'\1', text)
    text = re.sub(r'\s*<sup>\s*(\d+)\s*</sup>\s*', r'\1', text)
    return text

def get_parsed_text(parser, pdf_path, item_key, cache_dir):
    """Checks cache before triggering expensive PDF parsing."""
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = Path(cache_dir) / f"{item_key}.md"

    if cache_path.exists():
        return cache_path.read_text(encoding="utf-8")

    print(f"  -> [{MODE}] Parsing {item_key}...")
    text = parser.parse_to_text(pdf_path)
    cache_path.write_text(text, encoding="utf-8")
    return text

def process_item(db, parser, item):
    """Core logic: Parse -> Clean -> Chunk -> Embed -> Insert."""
    item_key = item['key']
    pdf_path = item['pdf_path']
    title = item['title'] or "Unknown Title"
    authors = item.get("authors", [])

    if pdf_path == "No PDF found":
        return False

    try:
        # 1. Parsing & Cleaning
        text = get_parsed_text(parser, pdf_path, item_key, PARSED_CACHE)
        text = clean_scientific_text(text)

        # 2. Hierarchical Chunking
        parents = parent_splitter.split_text(text)
        hierarchical_data = []

        for p_text in parents:
            child_texts = child_splitter.split_text(p_text)
            # Embedding only children for Vector Search
            child_vectors = embeddings_model.embed_documents(child_texts)

            hierarchical_data.append({
                'parent_text': p_text,
                'children_text': child_texts,
                'children_embeddings': child_vectors
            })

        # 3. Database Insertion
        db.insert_hierarchical_chunks(item_key, title, hierarchical_data, pdf_path, authors)
        return True
    except Exception as e:
        print(f"  ❌ Error indexing '{title}': {e}")
        return False

# --- 3. Main Execution Logic ---

def main():
    # --reset to force reset 
    parser = argparse.ArgumentParser(description="Zotero RAG Ingestion Pipeline")
    parser.add_argument("--reset", action="store_true", help="Wipe database and rebuild index")
    args = parser.parse_args()

    # Ensure directories exist
    DB_DIR.mkdir(parents=True, exist_ok=True)

    # Handle Reset
    if args.reset and DB_PATH.exists():
        print(f"⚠️ Reset flag detected. Removing {DB_PATH}...")
        DB_PATH.unlink()

    # Initialize components
    db = VectorDatabase(db_path=str(DB_PATH))
    zotero = ZoteroLocalClient(storage_path=str(ZOTERO_STORAGE))
    doc_parser = DocumentParser(use_gpu=USE_GPU, mode=MODE) # marker - gpu requires container

    print("🔍 Fetching Zotero library metadata...")
    library_items = zotero.get_library_metadata()
    indexed_keys = db.get_indexed_keys() # Get keys already in DB to skip

    # Filter items: if not resetting, only process what's missing
    to_process = [it for it in library_items 
        if it['key'] not in indexed_keys and it['pdf_path'] != "No PDF found"]

    if not to_process:
        print("✅ Database is up to date. No new papers found.")
        return

    print(f"🚀 Starting ingestion for {len(to_process)} newly papers...")

    # main loop over documents
    success_count = 0
    for i, item in enumerate(to_process):

        print(f"[{i+1}/{len(to_process)}] Processing: {item['title']}...")

        if process_item(db, doc_parser, item):
            success_count += 1

    if success_count > 0:
        print(f"🔄 {success_count} new papers found. Optimizing search index...")
        db.create_hnsw_index()
    else:
        print("zZZ No new papers to index.")

    print(f"\n✨ Ingestion Complete!")
    print(f"📊 Added {success_count} new papers to {DB_PATH}")

if __name__ == "__main__":
    main()
