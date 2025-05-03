from langchain_ollama import ChatOllama
from langchain_ollama.embeddings import OllamaEmbeddings

# Centralized Model Names
MODEL_NAME_SUPERVISOR = "hf.co/jedisct1/MiMo-7B-RL-GGUF:Q4_K_M"
MODEL_NAME_RETRIEVER = "qwen2.5:14b"
MODEL_NAME_GRAPH = "qwen2.5:14b"
MODEL_NAME_RAG = "qwen2.5:14b"
MODEL_NAME_EVALUATOR = "qwen2.5:14b" # Used for metrics calculation
MODEL_NAME_EMBEDDING = "nomic-embed-text:latest"

# Name used for tracking results in metrics files
# Corresponds to the model being evaluated (typically RAG/Graph/Evaluator)
EVALUATION_MODEL_NAME = MODEL_NAME_EVALUATOR

# --- Factory Functions for LLMs ---
def get_llm_supervisor():
    print(f"Instantiating Supervisor LLM: {MODEL_NAME_SUPERVISOR}")
    return ChatOllama(model=MODEL_NAME_SUPERVISOR, temperature=0)

def get_llm_retriever():
    print(f"Instantiating Retriever LLM: {MODEL_NAME_RETRIEVER}")
    return ChatOllama(model=MODEL_NAME_RETRIEVER, temperature=0.0)

def get_llm_graph():
    print(f"Instantiating Graph LLM: {MODEL_NAME_GRAPH}")
    return ChatOllama(model=MODEL_NAME_GRAPH, temperature=0.0)

def get_llm_rag():
    print(f"Instantiating RAG LLM: {MODEL_NAME_RAG}")
    return ChatOllama(model=MODEL_NAME_RAG, temperature=0.0)

def get_llm_evaluator():
    print(f"Instantiating Evaluator LLM: {MODEL_NAME_EVALUATOR}")
    return ChatOllama(model=MODEL_NAME_EVALUATOR, temperature=0.0)
# ------------------------------------

# Instantiate Embedding Model (Keep this instantiated, usually needed early)
embedding_model = OllamaEmbeddings(model=MODEL_NAME_EMBEDDING) 