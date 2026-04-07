from app.ingestion.database_schema import VectorDatabase
from app.retrieval.retriever import ResearchRetriever
from app.agent.researcher_modular import ResearchGenerator, ResearchCritic #, ReflectionLoop
from app.agent.researcher_graph import create_research_graph
from langchain_ollama import OllamaEmbeddings, ChatOllama
import yaml
from pathlib import Path

# for llm that requires api_keys
import os
from dotenv import load_dotenv

# load api_keys
load_dotenv()

def load_config():
    # go two levels up to the root where settings.yaml is located
    settings_dir = Path(__file__).parent.parent.parent.resolve()
    settings_file = settings_dir / "settings.yaml"
    with open(settings_file, "r") as f:
        return yaml.safe_load(f)


# Wraper to automatically pick the right class for model, ollama or commerical
def get_model(model_name, temperature):

    name = model_name.lower()

    # Switch based on model name or a 'provider' config key
    if "gemini" in name:
        from langchain_google_genai import ChatGoogleGenerativeAI
        print("gemini in use")
        return ChatGoogleGenerativeAI(
            model=model_name,
            temperature=temperature,
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )

    elif "gpt" in name or "openai" in name:
        from langchain_openai import ChatOpenAI
        print("openai in use")
        return ChatOpenAI(
            model=model_name,
            temperature=temperature,
            api_key=os.getenv("OPENAI_API_KEY")
        )

    else:
        from langchain_ollama import ChatOllama
        print("ollama in use")
        return ChatOllama(
            model=model_name,
            temperature=temperature
        )


def create_research_engine(overrides=None):
    """Initializes and returns the complete modular agent system."""

    cfg = load_config()
    # Apply Overrides from Streamlit if present
    if overrides:
        cfg['agent']['generator']['model'] = overrides['gen_model']
        cfg['agent']['generator']['temperature'] = overrides['gen_temp']
        cfg['agent']['critic']['model'] = overrides['crit_model']

    # Instantiate LLMs
    gen_llm = get_model(
        model_name = cfg['agent']['generator']['model'],
        temperature = cfg['agent']['generator']['temperature']
    )
    crit_llm = get_model(
        model_name = cfg['agent']['critic']['model'],
        temperature = cfg['agent']['critic']['temperature']
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
