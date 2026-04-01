import streamlit as st
import yaml
import re
import textwrap
import requests
from pathlib import Path
from app.utils.distill_query import distill_query

@st.cache_resource
def get_local_retriever():
    """Initializes the retriever once and keeps it in memory."""
    try:
        from app.core.config import retriever_module
        return retriever_module()
    except Exception as e:
        # Catching everything: ModuleNotFoundError, sqlite3.OperationalError (Locked), etc.
        #st.error(f"⚠️ Local Backend Unavailable: {e}")
        #print(e)
        return None

# --- Setup & Configuration ---
st.set_page_config(page_title="Zotero RAG Hub", layout="wide", page_icon="🔬")
st.markdown("""
    <style>
           /* Main content padding */
           .block-container {
                padding-top: 1rem;
                padding-bottom: 0rem;
                padding-left: 5rem;
                padding-right: 5rem;
            }

           /* Reduced Sidebar top margin/padding */
           [data-testid="stSidebarContent"] {
                padding-top: -5rem;
            }

           /* If you have a logo or top nav in the sidebar, this targets the very top */
           [data-testid="stSidebarNav"] {
                padding-top: -5rem;
                margin-top: -5rem;
            }
    </style>
    """, unsafe_allow_html=True)

if "messages" not in st.session_state:
    st.session_state.messages = []

# Get the directory where THIS script is located
# .parent gives the folder, .resolve() makes it an absolute path
current_dir = Path(__file__).parent.resolve()
settings_path = current_dir / "settings.yaml"
with open(settings_path, "r") as f:
    config_defaults = yaml.safe_load(f)

# Assuming cfg is your loaded settings.yaml
gen_llm_default = config_defaults['agent']['generator']['model']
gen_temp_default = config_defaults['agent']['generator']['temperature']
crit_llm_default = config_defaults['agent']['critic']['model']


def wrap_text(text, width=120):
    if not text: return ""
    wrapper = textwrap.TextWrapper(width=width, break_long_words=True, replace_whitespace=False)
    return "\n".join([wrapper.fill(line) for line in text.splitlines()])

# --- Sidebar: Connection & Settings ---
with st.sidebar:
    st.title("🤖 Agent Settings")

    # Toggle between Local logic and API logic
    conn_mode = st.radio("Connection Mode", ["Local (Direct Functions)", "Remote (FastAPI)"])
    use_api = (conn_mode == "Remote (FastAPI)")

    # Display a status badge for the DGX server if API is on
    if use_api:
        #st.info(f"⚡ Model configuration is managed by the server.")
        # Freeze the lists to only contain the YAML defaults
        display_gen_options = [gen_llm_default]
        display_crit_options = [crit_llm_default]
        gen_index = 0
        crit_index = 0
        status_text = "✨ Managed by DGX"
    else:
        LOCAL_GEN_OPTIONS = ["gpt-oss:20b", "llama3.1:8b"]
        LOCAL_CRIT_OPTIONS = ["nemotron-3-nano", "gpt-oss:120b"]
        # Use full lists and try to match the YAML default index
        display_gen_options = LOCAL_GEN_OPTIONS
        display_crit_options = LOCAL_CRIT_OPTIONS

        # Safely find indices for local mode defaults
        gen_index = display_gen_options.index(gen_llm_default) if gen_llm_default in display_gen_options else 0
        crit_index = display_crit_options.index(crit_llm_default) if crit_llm_default in display_crit_options else 0
        status_text = "🛠️ Local Configuration"

    with st.expander("Model Configuration", expanded=True):

        st.caption(status_text)
        gen_model = st.selectbox(
            "Generator", 
            options=display_gen_options, 
            index=gen_index,
            disabled=use_api,
            help="The primary LLM used for drafting research answers."
        )

        gen_temp = st.slider(
            "Gen Temperature", 
            0.0, 1.0, 
            value=float(gen_temp_default) if not use_api else float(gen_temp_default),
            disabled=use_api
        )

        crit_model = st.selectbox(
            "Critic", 
            options=display_crit_options, 
            index=crit_index,
            disabled=use_api
        )

        max_retries = st.slider("Max Tries (reflection)", 0, 5, 0)

    with st.expander("Search Filters", expanded=True):
        search_mode = st.radio("Mode", ["Semantic Search Only", "Research Assistant (LLM)"])
        search_type = st.radio("Search Type", ["standard", "hierarchical"])

        # Dynamic title fetching based on mode
        available_titles = []
        if use_api:
            try:
                response = requests.get("http://localhost:8000/metadata/titles", timeout=2)
                available_titles = [t[0] for t in response.json()] if response.status_code == 200 else []
            except:
                st.warning("Could not connect to Remote API. Is the server running?")
        else:
            # Attempt to use local retriever
            local_retriever = get_local_retriever()
            if local_retriever:
                try:
                    # Wrap the actual DB query in case of "Database Locked"
                    titles = local_retriever.db.get_connection().execute("SELECT DISTINCT title FROM paper_chunks").fetchall()
                    available_titles = [t[0] for t in titles]
                except Exception as e:
                    st.error("Storage Error: Database is likely locked by another process.")
            else:
                st.info("Please switch to 'Remote Mode' if you don't have local dependencies installed.")

        selected_titles = st.multiselect("Filter by Paper(s)", options=available_titles)
        top_k = st.slider("Context Chunks (k)", 1, 12, 5)
        window = st.slider("# Neighboring Chunks", 0, 5, 2)

    if st.button("🗑️ Clear Chat", type="primary"):
        st.session_state.messages = []
        st.rerun()

# --- Main Logic Execution ---
st.title("🔬 Scientific Research Assistant")

# Display History
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("sources"):
            with st.expander("📚 View Cited Sources"):
                st.markdown(msg["sources"])

# User Input
if prompt := st.chat_input("Search your library..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.status("Analyzing...", expanded=True) as status:
            
            if use_api:
                # --- REMOTE API EXECUTION ---
                payload = {
                    "question": prompt,
                    "top_k": top_k,
                    "window": window,
                    "search_mode": search_mode,
                    "mode": search_type,
                    "metadata_filters": {"title": selected_titles} if selected_titles else None,
                    "max_retries": max_retries
                }
                try:
                    res = requests.post("http://localhost:8000/research", json=payload)
                    res.raise_for_status()
                    data = res.json()
                    answer, verified, feedback = data["answer"], data["verified"], data["feedback"]
                    context_map = data["context_map"]
                except Exception as e:
                    st.error(f"API Error: {e}"); st.stop()
            else:
                # create_research_engine needs to be reloaded because of the input parameters
                from app.core.config import create_research_engine
                # --- LOCAL FUNCTION EXECUTION ---
                clean_query = distill_query(prompt)
                #clean_query = f"Represent this sentence for searching relevant passages: {clean_query}"

                context_text, _, context_map = local_retriever.get_relevant_context(
                    clean_query, top_k=top_k, window=window, 
                    filter_dict={"title": selected_titles} if selected_titles else None, mode=search_type
                )

                if search_mode == "Semantic Search Only":
                    answer, verified, feedback = "Relevant sections found:", True, ""
                else:

                    config_overrides = {
                        "gen_model": gen_model,
                        "gen_temp": gen_temp,
                        "crit_model": crit_model,
                        "max_retries": max_retries,
                    }

                    app, _ = create_research_engine(overrides=config_overrides)

                    # langGraph
                    # ResearchState(TypedDict):
                    initial_input = {
                        "question": prompt,
                        "context": context_text,
                        "iterations": 0,
                        "verified": False,
                        "feedback": "",
                        "max_retries": max_retries,
                    }

                    # Invoke the graph (returns the final dictionary of the State)
                    final_state = app.invoke(initial_input)
                    # Extract the specific data you need
                    answer = final_state["draft"]
                    verified = final_state["verified"]
                    feedback = final_state["feedback"]

            # --- SHARED POST-PROCESSING (Citations) ---
            if search_mode == "Semantic Search Only":
                cited_context = "".join([f"\n{'-'*30}\n{tag}\n{'-'*30}\n{text}" for tag, text in context_map.items()])
            else:
                cited_tags = list(set(re.findall(r'([\[【(]Source\d+[\]】)])', answer)))
                cited_context = "".join([f"**{c}**:\n{context_map.get(c, '')}\n\n" for c in cited_tags])

            status.update(label="Complete", state="complete")

        # --- UI Rendering ---
        #badge = "🟢 Semantic Search" if search_mode == "Semantic Search Only" else ("✅ **Verified**" if verified else "⚠️ **Unverified**")
        #if search_mode != "Semantic Search Only" and max_retries == 0: badge = "🔵 No critic"

        # Render UI components (badges, text, and expanders)
        if search_mode=="Semantic Search Only":
            badge = "🟢 Semantic Search"
        elif max_retries==0:
            badge = "🔵 No critic:"
        elif verified:
            badge = "✅ **Verified**"
        else:
            badge = "⚠️ **Unverified**"

        full_response = f"{badge}\n\n{answer}"
        st.markdown(full_response)

        if cited_context:
            with st.expander("📚 View Cited Sources", expanded=(search_mode == "Semantic Search Only")):
                st.markdown(cited_context)
                with st.expander("📚 View Cited Sources"):
                    st.code(wrap_text(cited_context), language=None)
        if feedback:
            with st.expander("📝 View Critic Feedback"):
                #st.code(wrap_text(feedback))
                st.markdown(feedback)

        st.session_state.messages.append({"role": "assistant", "content": full_response, "sources": cited_context})
