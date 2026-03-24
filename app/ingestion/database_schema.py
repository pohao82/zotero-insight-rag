import duckdb
import os
from pathlib import Path
from typing import List, Optional, Set, Dict, Any, Union
from pydantic import BaseModel, Field

# --- 1. The Data Contract (Systematic Schema Management) ---
class PaperChunk(BaseModel):
    """Defines exactly what a chunk looks like before it hits the DB."""
    item_key: str
    title: str
    author: List[str] =[]  # New Field
    text: str
    embedding: Optional[List[float]] = None
    file_path: str
    chunk_index: int
    parent_id: Optional[int] = None
    is_child: bool = False
    tags: str = ""

# --- 2. The Database Class ---
class VectorDatabase:
    def __init__(self, db_path: Optional[str] = None) -> None:
        if db_path:
            self.db_path = db_path
            # --- NEW: Ensure the parent directory exists ---
            db_dir = os.path.dirname(self.db_path)
            if db_dir and not os.path.exists(db_dir):
                print(f"📁 Creating missing directory: {db_dir}")
                os.makedirs(db_dir, exist_ok=True)
        # -----------------------------------------------
        else:
            # 1. Find where THIS file (database.py) is
            this_file = Path(__file__).resolve()
            # 2. Go up 3 levels: ingestion -> app -> project_root
            project_root = this_file.parent.parent.parent
            self.db_path = str(project_root / "data" / "zotero_lieb_md.db")

        # Ensure the directory exists
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)

        print(f"Connecting to database at: {self.db_path}")
        # con = duckdb.connect(':memory:')
        # print(type(con)) 
        # Output: <class 'duckdb.duckdb.DuckDBPyConnection'>
        # Using the alias duckdb.DuckDBPyConnection is standard 
        self.con: duckdb.DuckDBPyConnection = duckdb.connect(self.db_path)
        self._initialize_vss()
        self._create_or_update_table()

    def _initialize_vss(self) -> None:
        """Initializes the Vector Similarity Search extension."""
        self.con.execute("INSTALL vss; LOAD vss;")

        ## FTS For hybrid search
        ## -----------------------------------------
        #self.con.execute("INSTALL fts; LOAD fts;")
        #try:
        #    self.con.execute("PRAGMA drop_fts_index('paper_chunks')")
        #except:
        #    pass # Index didn't exist, which is fine
        #self.con.execute("PRAGMA create_fts_index('paper_chunks', 'rowid', 'text')")
        ## -----------------------------------------

        # Mandatory for saving HNSW indexes to a .db file
        self.con.execute("SET hnsw_enable_experimental_persistence = true;")

    # create_table
    def _create_or_update_table(self) -> None:
        """Dynamically creates the table based on PaperChunk fields."""
        type_map = {
            str: "TEXT",
            int: "INTEGER",
            bool: "BOOLEAN",
            Optional[int]: "INTEGER",
            Optional[List[float]]: "FLOAT[1024]" # Custom vector type
        }

        # Build columns from Pydantic model fields
        cols = []
        for name, field in PaperChunk.model_fields.items():
            db_type = type_map.get(field.annotation, "TEXT")
            cols.append(f"{name} {db_type}")

        # Using 'CREATE TABLE IF NOT EXISTS' for initial setup
        query = f"CREATE TABLE IF NOT EXISTS paper_chunks ({', '.join(cols)})"
        self.con.execute(query)


    def create_hnsw_index(self):
        """Call this AFTER all papers are ingested for maximum efficiency."""
        print("🚀 Building optimized HNSW index (Bulk Mode)...")
        # Drop if exists to ensure a clean, compact build to avoid repeated indexing
        self.con.execute("DROP INDEX IF EXISTS hnsw_idx")

        self.con.execute("""
            CREATE INDEX hnsw_idx ON paper_chunks 
            USING HNSW (embedding) 
            WITH (metric = 'cosine', M = 8, ef_construction = 120);
        """)
        self.con.execute("CHECKPOINT;")
        print("✅ Index built and compacted.")


    def _execute_insert(self, chunk: PaperChunk) -> None:
        # create dynamtically
        """Inserts data using positional parameters to avoid parser errors."""
        data = chunk.model_dump() 

        # 1. Get the keys and values in a consistent order
        columns = ", ".join(data.keys())
        # 2. generate equal number of "?" for the query place holder
        placeholders = ", ".join(["?" for _ in data.keys()])
        # 3. Get the values as a list/tuple in that same order
        values = list(data.values())

        #"INSERT INTO paper_chunks (key1, key2, ..) VALUES (?,?, ..)"
        sql = f"INSERT INTO paper_chunks ({columns}) VALUES ({placeholders})"

        self.con.execute(sql, values)


    def insert_hierarchical_chunks(
        self,
        item_key: str,
        title: str,
        parent_data: List[Dict[str, Any]],
        pdf_path: Union[str, Path],
        authors: List[str]=[], 
        tags: str = ""
    ) -> None:
        """
        Systematically inserts hierarchical chunks using Pydantic for validation.

        Expects parent_data as a list of dicts: 
        [{
          'parent_text': str, # one parent chunk
          'children_text': [str], # all the children belong to the same parent   
          'children_embeddings': [[float]] # all the children vectors 
         },...]
         Each dict corresponds to one parent chunk and all its children chunks
        """
        global_index = 0
        # loop over parent chunks
        for p_idx, p_block in enumerate(parent_data):

            # Create a Pydantic object for the Parent
            parent = PaperChunk(
                item_key=item_key,
                title=title,
                author = authors,    # New Field
                text=p_block['parent_text'],
                embedding=None, # Parents dont have a vector
                file_path=str(pdf_path),
                chunk_index=global_index,
                parent_id=p_idx,
                is_child=False,
                tags=tags
            )
            self._execute_insert(parent)
            global_index += 1

            # loop over children
            for c_text, c_vector in zip(p_block['children_text'], p_block['children_embeddings']):
                # Create Pydantic objects for Children
                child = PaperChunk(
                    item_key=item_key,
                    title=title,
                    author=authors,    # New Field
                    text=c_text,
                    embedding=c_vector,
                    file_path=str(pdf_path),
                    chunk_index=global_index,
                    parent_id=p_idx,
                    is_child=True,
                    tags=tags
                )
                self._execute_insert(child)
                global_index += 1

        self.con.execute("CHECKPOINT;")

    def get_connection(self) -> duckdb.DuckDBPyConnection:
        """Returns the raw connection if custom SQL is needed."""
        return self.con

    def close(self) -> None:
        self.con.close()

    # used to recover interruptions
    def get_indexed_keys(self) -> Set[str]:
        result = self.con.execute("SELECT DISTINCT item_key FROM paper_chunks").fetchall()
        return {row[0] for row in result}
