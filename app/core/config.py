from app.ingestion.database_schema import VectorDatabase
from app.retrieval.retriever import ResearchRetriever
from app.agent.researcher_modular import ResearchGenerator, ResearchCritic #, ReflectionLoop
from app.agent.researcher_graph import create_research_graph
from langchain_ollama import OllamaEmbeddings, ChatOllama
import yaml
from pathlib import Path

def load_config():
    # go two levels up to the root where settings.yaml is located
    settings_dir = Path(__file__).parent.parent.parent.resolve()
    settings_file = settings_dir / "settings.yaml"
    with open(settings_file, "r") as f:
        return yaml.safe_load(f)

def create_research_engine(overrides=None):
    """Initializes and returns the complete modular agent system."""

    cfg = load_config()
    # Apply Overrides from Streamlit if present
    if overrides:
        cfg['agent']['generator']['model'] = overrides['gen_model']
        cfg['agent']['generator']['temperature'] = overrides['gen_temp']
        cfg['agent']['critic']['model'] = overrides['crit_model']

    # Models
    gen_llm = ChatOllama(
        model=cfg['agent']['generator']['model'], 
        temperature=cfg['agent']['generator']['temperature']
    )
    crit_llm = ChatOllama(
        model=cfg['agent']['critic']['model'], 
        temperature=cfg['agent']['critic']['temperature']
    )

    # Agent Components
    generator = ResearchGenerator(gen_llm)
    critic = ResearchCritic(crit_llm)

    # Replace ReflectionLoop with LangGraph
    graph_app = create_research_graph(generator, critic)
    return graph_app, generator


def retriever_module(db_path=None, embedding_model=None):

    # VDB refrieval infrastructure
    cfg = load_config()

    if not db_path:
        db_path = cfg['infrastructure']['db_path']
    if not embedding_model:
        embedding_model = cfg['infrastructure']['embedding_model']

    db = VectorDatabase(db_path)
    embeddings = OllamaEmbeddings(model=embedding_model)
    retriever = ResearchRetriever(db, embeddings)

    return retriever 
