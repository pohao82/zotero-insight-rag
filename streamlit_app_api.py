import streamlit as st
import yaml
import re
import textwrap
from pathlib import Path
import requests
from app.utils.distill_query import distill_query

st.set_page_config(page_title="Zotero RAG Hub", layout="wide", page_icon="🔬")
st.markdown("""
    <style>
           .block-container {
                padding-top: 1rem;
                padding-bottom: 0rem;
                padding-left: 5rem;
                padding-right: 5rem;
            }
    </style>
    """, unsafe_allow_html=True)

# --- 1. Load Defaults and Database Titles ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# Get the directory where THIS script is located
# .parent gives the folder, .resolve() makes it an absolute path
current_dir = Path(__file__).parent.resolve()
settings_path = current_dir / "settings.yaml"
with open(settings_path, "r") as f:
    config_defaults = yaml.safe_load(f)

# Helper for wrapping
def wrap_text(text, width=120):
    if not text: return ""
    wrapper = textwrap.TextWrapper(width=width, break_long_words=True, replace_whitespace=False)
    return "\n".join([wrapper.fill(line) for line in text.splitlines()])


# request for titles
try:
    response = requests.get("http://localhost:8000/metadata/titles")
    available_titles = response.json() if response.status_code == 200 else []
except:
    available_titles = [] # Fallback if API isn't up yet


#available_authors = retriever.db.get_connection().execute("SELECT DISTINCT author FROM paper_chunks").fetchall()
available_titles = [t[0] for t in available_titles]

# --- 2. Sidebar ---
with st.sidebar:
    st.title("🤖 Agent Settings")

    with st.expander("Model Configuration", expanded=True):
        gen_model = st.selectbox("Generator (Drafting)", 
                                 ["gpt-oss:20b", "llama3:8b"], 
                                 index=0)
        gen_temp = st.slider("Gen Temperature", 0.0, 1.0, 0.1)
        crit_model = st.selectbox("Critic (Verification)", 
                                  ["nemotron-3-nano", "gpt-oss:120b"], 
                                  index=0)

        max_retries = st.slider("Max Tries(reflection)", 0, 5, 0)

    with st.expander("Search Filters", expanded=True):
        search_mode = st.radio("Mode", ["Semantic Search Only", "Research Assistant (LLM)" ])
        search_type = st.radio("Search Type", ["standard", "hierarchical"])
        selected_titles = st.multiselect("Filter by Paper(s)", options=available_titles)
        top_k = st.slider("Context Chunks (k)", 1, 12, 5)
        window = st.slider("# Neighboring Chunks", 0, 5, 2)

    if st.button("🗑️ Clear Chat", type="primary"):
        st.session_state.messages = []
        st.rerun()

    config_overrides = {
        "gen_model": gen_model,
        "gen_temp": gen_temp,
        "crit_model": crit_model,
        "max_retries": max_retries
    }

    metadata_filters = {"title": selected_titles} if selected_titles else None

# --- 3. Main Chat UI ---
st.title("🔬 Scientific Research Assistant")

# Display History - Using structured rendering
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        # Only show expanders for assistant messages that have sources
        if msg["role"] == "assistant" and "sources" in msg:
            is_semantic = msg.get("mode") == "Semantic Search Only"
            with st.expander("📚 View Cited Sources", expanded=is_semantic):
                st.code(wrap_text(msg["sources"]), language=None)

# --- 4. Execution ---
if prompt := st.chat_input("Search your library..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.status("Requesting Analysis from Backend...", expanded=True) as status:

            # Prepare API Payload
            payload = {
                "question": prompt,
                "top_k": top_k,
                "window": window,
                "search_mode": search_mode,
                "mode": search_type,
                "metadata_filters": metadata_filters,
                "max_retries": max_retries
            }

            try:
                # Call the FastAPI endpoint
                response = requests.post("http://localhost:8000/research", json=payload)
                response.raise_for_status()
                data = response.json()

                # Extract data from API response
                answer = data["answer"]
                verified = data["verified"]
                feedback = data["feedback"]
                sources = data["sources"]
                context_map = data["context_map"]

                if search_mode == "Semantic Search Only":
                    cited_context = ""
                    for source_tag in context_map:
                        cited_context += '\n\n'+'-'*70 
                        cited_context += f'\n{source_tag}\n'
                        cited_context += '-'*70 
                        cited_context += context_map[source_tag]
                else:
                    # Handle Citations
                    regex = r'([\[【(]Source\d+[\]】)])'
                    cited_tags = re.findall(regex, answer)
                    cited_tags = list(set(cited_tags))
                    cited_context = "".join([f"{c}:\n{context_map.get(c, '')}\n\n" for c in cited_tags])

                status.update(label="Analysis Complete", state="complete")

            except Exception as e:
                st.error(f"API Error: {str(e)}")
                status.update(label="Error", state="error")
                st.stop()

        # Render UI components (badges, text, and expanders)
        if search_mode=="Semantic Search Only":
            badge = "🟢 Semantic Search"
        elif max_retries==0:
            badge = "🔵 No critic:"
        elif verified:
            badge = "✅ **Verified**"
        else:
            badge = "⚠️ **Unverified**"

        display_text = f"{badge}\n\n{answer}"
        st.markdown(display_text)

        if cited_context:
            should_expand = (search_mode == "Semantic Search Only")
            with st.expander("📚 View Cited Sources",expanded=should_expand):
                st.code(wrap_text(cited_context), language=None)

        # from critic
        if feedback:
            with st.expander("📚 View Feedback"):
                st.code(wrap_text(feedback), language=None)

        # Save to session history
        st.session_state.messages.append({
            "role": "assistant", 
            "content": display_text,
            "sources": cited_context
        })
