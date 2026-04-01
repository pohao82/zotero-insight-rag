import duckdb
from langchain_core.messages import HumanMessage, SystemMessage

class ResearchRetriever:
    def __init__(self, db_manager, embeddings_model, llm_model=None):
        """
        db_manager: An instance VectorDatabase class
        embeddings_model: LangChain OllamaEmbeddings instance
        llm_model: LangChain ChatOllama instance (optional, for synthesis)
        """
        self.db = db_manager
        self.embeddings = embeddings_model
        #self.llm = llm_model

    def get_relevant_context(self, question, top_k=5, window=1, filter_dict=None, mode='standard'):
        """
        question: User query string
        top_k: Number of relevant chunks to retrieve
        window: Number of neighbor chunks to include
        filter_dict: Dictionary of metadata filters (e.g., {'title': 'Nature Physics 2024'})
        mode: 'standard' (original +/- N chunks) or 'hierarchical' (Parent-Child)
        """
        # Convert question string to embedding vector
        query_vector = self.embeddings.embed_query(question)
        # Connect to vectordatabase
        con = self.db.get_connection()

        # -----------------------
        #   Constructing query
        # -----------------------
        # Base query changes based on mode
        where_clauses = [] # WHERE condition in SQL, base filter
        params = []

        #if mode == 'hierarchical':
        #    where_clauses.append("is_child = TRUE")
        where_clauses.append("is_child = TRUE")

        # Add additional metadata filters
        if filter_dict:
            for key, value in filter_dict.items():
                # check if value is a list
                if isinstance(value, list) and len(value) > 0:
                    # Handle multiple selections (e.g., multiselect in Streamlit)
                    placeholders = ", ".join(["?"] * len(value)) # how many values = how many "?"
                    where_clauses.append(f"{key} IN ({placeholders})")
                    params.extend(value)
                # if value is a single element, standard
                elif value:
                    where_clauses.append(f"{key} = ?")
                    params.append(value)

        where_str = " WHERE " + " AND ".join(where_clauses) if where_clauses else ""

        # Need chunk_index and parent_id to find neighbors/parents later
        #query = f"""
        #    SELECT title, item_key, chunk_index, parent_id, is_child
        #    FROM paper_chunks
        #    {where_str}
        #    ORDER BY array_cosine_distance(embedding, ?::FLOAT[1024]) ASC
        #    LIMIT {top_k}
        #"""
        query = f"""
            SELECT 
                title, item_key, chunk_index, parent_id,
                array_cosine_distance(embedding, ?::FLOAT[1024]) AS distance
            FROM paper_chunks
            {where_str}
            ORDER BY distance ASC
            LIMIT {top_k}
        """
        # return a list of rows that contains (title, item_key, chunk_index, parent_id)

        # Re-append query_vector for the distance calculation in the ORDER BY clause
        params.append(query_vector)

        ##-----------------------------------------
        ## Query 1: Unfiltered
        #plan_unfiltered = con.execute("""
        #    EXPLAIN SELECT title, array_cosine_distance(embedding, ?::FLOAT[1024]) AS distance
        #    FROM paper_chunks
        #    ORDER BY distance ASC LIMIT 5
        #""", params).fetchone()[1]

        ## Query 2: Filtered
        #plan_filtered = con.execute("""
        #    EXPLAIN SELECT title, array_cosine_distance(embedding, ?::FLOAT[1024]) AS distance
        #    FROM paper_chunks
        #    WHERE is_child = TRUE
        #    ORDER BY distance ASC LIMIT 5
        #""", params).fetchone()[1]

        #print("--- Unfiltered Plan ---")
        #print(plan_unfiltered)
        #print("\n--- Filtered Plan ---")
        #print(plan_filtered)

        #print(query)
        ##-----------------------------------------

        # rel_docs are the matched (relevant) vectors (all children)
        rel_docs = con.execute(query, params).fetchall()

        context_text = "" # full retrieved results as one big string
        context_map = {}  # dict for source - context mapping
        sources = set()   # tract distinct titles

        # To avoid "brutal overlap," keep track of what we've already added
        seen_parents = set()

        # All vectors are children, depend on the mode to decide whether parents are return
        for i, row in enumerate(rel_docs):
            title, item_key, chunk_index, parent_id, dist = row
            source_tag = f"[Source{i+1}]"
            #print(f'chunck_id {chunk_index}, dist: {dist}')

            # parent-child
            if mode == 'hierarchical':
                # Skip if already pulled this parent section for a previous hit
                if (item_key, parent_id) in seen_parents:
                    continue

                # Fetch the Parent + Adjacent Parents (Hybrid Strategy)
                expanded_text = self.get_hierarchical_context(
                    item_key, parent_id, window=window
                )

                seen_parents.add((item_key, parent_id))

            # simply children chunk with neighbors
            else:
                # Original Standard Logic: +/- N Neighboring Chunks
                expanded_text = self.get_child_neighbors(
                    item_key, chunk_index, window
                )

            # Label it clearly for the Generator llm
            context_text += f"\n--- {source_tag} ---\n**TITLE**: {title}\n\n**CONTENT**:\n\n{expanded_text}\n"
            sources.add(title)

            expanded_text = f"\n**TITLE:** {title}\n\n**CONTENT**:\n{expanded_text}\n"
            context_map[source_tag] = expanded_text

        return context_text, list(sources), context_map


    def get_hierarchical_context(self, item_key, parent_id, window=0):
        """
        Fetches the target Parent block plus +/- N neighboring Parent blocks.
        """
        # We use parent_id as the index for logical sections
        query = """
            SELECT parent_id, text FROM paper_chunks
            WHERE item_key = ?
            AND is_child = FALSE
            AND parent_id BETWEEN ? AND ?
            ORDER BY parent_id ASC
        """
        results = self.db.con.execute(query, [
            item_key,
            parent_id - window,
            parent_id + window
        ]).fetchall()

        # Parents are usually distinct paragraphs/sections,
        # join them with double newlines
        return "\n\n".join([r[1] for r in results])


    def get_child_neighbors(self, item_key, chunk_index, window=1, overlap_len=100):
        """
        Retrieves a specific child chunk and its +/- neighbors,
        automatically skipping any interleaved parent rows.
        """
        query = """
            WITH ChildSequence AS (  -- Map child chunk_index to a new local sequence
                SELECT
                    chunk_index,
                    text,
                    ROW_NUMBER() OVER (ORDER BY chunk_index ASC) as child_rank
                FROM paper_chunks
                WHERE item_key = ? AND is_child = TRUE
            )
            SELECT text FROM ChildSequence
            WHERE child_rank BETWEEN
                (SELECT child_rank FROM ChildSequence WHERE chunk_index = ?) - ?
                AND
                (SELECT child_rank FROM ChildSequence WHERE chunk_index = ?) + ?
            ORDER BY chunk_index ASC
        """

        params = [item_key, chunk_index, window, chunk_index, window]
        results = self.db.con.execute(query, params).fetchall()

        if not results:
            return ""

        # begin with first chunk
        merged_text = results[0][0]

        for i in range(1, len(results)):
            current_text = results[i][0]

            # make sure the text is long enough, to avoid accidental overlap
            anchor = current_text[:overlap_len//2]
            search_area = merged_text[round(-overlap_len*1.5):] # last portion of the previous text chunck
            anchor_pos = search_area.find(anchor)

            # remove overlap
            if anchor_pos != -1:
                overlap_actual_len = len(search_area) - anchor_pos
                merged_text += current_text[overlap_actual_len:]
            else:
                merged_text += "\n" + current_text

        return merged_text

