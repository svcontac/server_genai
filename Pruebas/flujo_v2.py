from langgraph_supervisor import create_supervisor
from langgraph_supervisor import create_handoff_tool
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore
from langchain_ollama import ChatOllama
from IPython.display import Image, display
import pandas as pd
import plotly.express as px
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_core.documents import Document
import os
import uuid
import shutil # Added for index deletion
import json # Added for manifest handling
from ragas.integrations.langgraph import convert_to_ragas_messages
from ragas.metrics import ToolCallAccuracy
from ragas.dataset_schema import MultiTurnSample
import ragas.messages as r

# Import models from the central file
from models import (
    get_llm_supervisor, get_llm_retriever, get_llm_graph, get_llm_rag, # Import factory functions
    embedding_model
)

# model_name = "MFDoom/deepseek-r1-tool-calling:14b"
# model_name = "PetrosStav/gemma3-tools:12b"
# model_name = 'qwen2.5:14b'
# model_rag = ChatOllama(model=model_name, temperature=0.0)
# model_recuperador = ChatOllama(model="deepseek-coder:14b", temperature=0.0)

# model_supervisor = ChatOllama(model="qwen2.5:14b", temperature=0)

# Checkpointer and Store remain here
checkpointer = InMemorySaver()
store = InMemoryStore()

# Define embedding model and FAISS path
# embedding_model = OllamaEmbeddings(model="nomic-embed-text") # Removed definition
FAISS_INDEX_PATH = "faiss_index"
FAISS_MANIFEST_PATH = "faiss_index_manifest.json" # Path for the manifest file

#Tools

def plot_line(file_path: str, x_col: str | int, y_col: str | int) -> None:
    """Create a line plot from an Excel/CSV file using specified x and y columns.
    
    Args:
        file_path: Path to the Excel (.xls/.xlsx) or CSV file
        x_col: Name (str) or index (int) of the column to use for x-axis
        y_col: Name (str) or index (int) of the column to use for y-axis
    """
    
    # Read the file based on extension
    if file_path.endswith(('.xls', '.xlsx')):
        df = pd.read_excel(file_path)
    elif file_path.endswith('.csv'):
        df = pd.read_csv(file_path)
    else:
        raise ValueError("File must be Excel (.xls/.xlsx) or CSV format")
        
    # Determine column names based on input type
    x_col_name = df.columns[x_col] if isinstance(x_col, int) else x_col
    y_col_name = df.columns[y_col] if isinstance(y_col, int) else y_col

    # Select data based on input type
    x_data = df.iloc[:, x_col] if isinstance(x_col, int) else df[x_col]
    y_data = df.iloc[:, y_col] if isinstance(y_col, int) else df[y_col]

    # Create the line plot using plotly
    fig = px.line(df, x=x_col_name, y=y_col_name, 
                  title=f'{y_col_name} vs {x_col_name}')
    
    fig.update_layout(
        xaxis_title=x_col_name,
        yaxis_title=y_col_name,
        xaxis=dict(showgrid=True),
        yaxis=dict(showgrid=True)
    )
    
    fig.show()
    return "Gráfico de línea (plot_line) realizado exitosamente"

def plot_histogram(file_path: str, column: str | int) -> None:
    """Create a histogram plot from an Excel/CSV file using specified column.
    
    Args:
        file_path: Path to the Excel (.xls/.xlsx) or CSV file
        column: Name (str) or index (int) of the column to plot histogram for
    """
    
    # Read the file based on extension
    if file_path.endswith(('.xls', '.xlsx')):
        df = pd.read_excel(file_path)
    elif file_path.endswith('.csv'):
        df = pd.read_csv(file_path)
    else:
        raise ValueError("File must be Excel (.xls/.xlsx) or CSV format")
        
    # Determine column name based on input type
    col_name = df.columns[column] if isinstance(column, int) else column

    # Select data based on input type
    data = df.iloc[:, column] if isinstance(column, int) else df[column]

    # Create the histogram using plotly
    fig = px.histogram(df, x=col_name,
                      title=f'Histogram of {col_name}')
    
    fig.update_layout(
        xaxis_title=col_name,
        yaxis_title='Count',
        xaxis=dict(showgrid=True),
        yaxis=dict(showgrid=True)
    )
    
    fig.show()
    return "Histograma (plot_histogram) realizado exitosamente"

# --- RAG Functions ---

def get_embedding(query: str) -> list[float]:
    """Generates embeddings for the given query using the Ollama embedding model."""
    return embedding_model.embed_query(query)

def process_and_store_pdfs(
    pdf_paths: list[str],
    save_path: str = FAISS_INDEX_PATH,
    chunk_size: int = 512,  # Reduced chunk size
    chunk_overlap: int = 100 # Reduced chunk overlap
) -> None:
    """
    Loads PDFs, splits them into chunks, generates embeddings,
    and stores them in a FAISS vector store, only if necessary.

    Checks a manifest file to determine if the index needs rebuilding based on
    the provided pdf_paths.

    Args:
        pdf_paths: List of paths to PDF files.
        save_path: Path to save the FAISS index.
        chunk_size: Size of text chunks.
        chunk_overlap: Overlap between text chunks.
    """

    # --- Check if index is up-to-date using manifest --- 
    current_abs_paths = sorted([os.path.abspath(p) for p in pdf_paths])
    rebuild_needed = True # Assume rebuild needed unless proven otherwise

    if os.path.exists(save_path) and os.path.exists(FAISS_MANIFEST_PATH):
        try:
            with open(FAISS_MANIFEST_PATH, 'r') as f:
                manifest_data = json.load(f)
                previous_abs_paths = sorted(manifest_data.get('processed_files', []))
            
            if current_abs_paths == previous_abs_paths:
                print(f"FAISS index at '{save_path}' is up-to-date with the specified PDF files. Skipping rebuild.")
                rebuild_needed = False
            else:
                print("PDF file list has changed. Rebuilding FAISS index.")
        except json.JSONDecodeError:
            print(f"Error reading manifest file '{FAISS_MANIFEST_PATH}'. Assuming rebuild is needed.")
        except Exception as e:
            print(f"Error checking manifest file: {e}. Assuming rebuild is needed.")
    else:
        print(f"FAISS index or manifest file not found. Building index.")

    if not rebuild_needed:
        return # Exit the function if no rebuild is necessary
    # -----------------------------------------------------

    # --- Proceed with rebuild if needed ---
    print("Starting PDF processing and index build...")

    # --- Delete existing index and manifest if they exist ---
    if os.path.exists(save_path):
        print(f"Deleting existing FAISS index at {save_path}...")
        try:
            shutil.rmtree(save_path)
            print("Existing index deleted.")
        except OSError as e:
            print(f"Error deleting existing index: {e}. Attempting to continue...")
            # If deletion fails, we might overwrite, which is often okay for FAISS
    if os.path.exists(FAISS_MANIFEST_PATH):
         try:
             os.remove(FAISS_MANIFEST_PATH)
             print("Existing manifest file deleted.")
         except OSError as e:
             print(f"Error deleting existing manifest: {e}.")
    # --------------------------------------------------------

    all_docs = []
    loaded_paths_for_manifest = [] # Keep track of successfully loaded paths
    for pdf_path in pdf_paths:
        abs_path = os.path.abspath(pdf_path)
        if os.path.exists(pdf_path):
            try:
                loader = PyPDFLoader(pdf_path)
                docs = loader.load()
                all_docs.extend(docs)
                loaded_paths_for_manifest.append(abs_path) # Add to list for manifest
                print(f"Successfully loaded: {pdf_path}")
            except Exception as e:
                 print(f"Warning: Failed to load {pdf_path}. Error: {e}. Skipping this file.")
        else:
            print(f"Warning: PDF file not found at {pdf_path}. Skipping.")

    if not all_docs:
        print("No documents successfully loaded. FAISS index cannot be created.")
        return

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    split_docs = text_splitter.split_documents(all_docs)

    if not split_docs:
        print("No chunks created from documents. FAISS index will not be created.")
        return

    print(f"Creating FAISS index from {len(split_docs)} chunks with chunk_size={chunk_size}, chunk_overlap={chunk_overlap}...")
    try:
        vectorstore = FAISS.from_documents(split_docs, embedding_model)
        vectorstore.save_local(save_path)
        print(f"FAISS index saved successfully to {save_path}")

        # --- Update manifest file after successful save --- 
        try:
            manifest_content = {'processed_files': sorted(loaded_paths_for_manifest)}
            with open(FAISS_MANIFEST_PATH, 'w') as f:
                json.dump(manifest_content, f, indent=4)
            print(f"Manifest file updated at '{FAISS_MANIFEST_PATH}'")
        except Exception as e:
            print(f"Error writing manifest file: {e}")
        # ---------------------------------------------------

    except Exception as e:
         print(f"Error creating or saving FAISS index: {e}")
         # Attempt to clean up potentially corrupt index/manifest if save failed
         if os.path.exists(save_path):
             try: shutil.rmtree(save_path) 
             except: pass
         if os.path.exists(FAISS_MANIFEST_PATH):
             try: os.remove(FAISS_MANIFEST_PATH) 
             except: pass

def retrieve_file(query: str) -> str:
    """
    Retrieve the main file from the FAISS vector store based on the query.
 
    Args:
        query: The search query to find the main file.
 
    Returns:
        str: string with the name of the recovered file
    """
    if not os.path.exists(FAISS_INDEX_PATH):
        return "Error: FAISS index not found. Please run `process_and_store_pdfs` first."
 
    try:
        # Added allow_dangerous_deserialization=True
        vectorstore = FAISS.load_local(FAISS_INDEX_PATH, embedding_model, allow_dangerous_deserialization=True)
        # No need to call get_embedding separately, similarity_search handles it
        results: list[Document] = vectorstore.similarity_search(query, k=1)
 
        if not results:
            return "No relevant context found for the query."
 
        # Path del archivo recuperado
        main_file = results[0].metadata['source']
 
        return f"{main_file}"
 
    except Exception as e:
        return f"Error retrieving context from FAISS: {e}"
   
# Modify retrieve_context to use the vector store
def retrieve_context(query: str, k: int = 5) -> str:
    """
    Retrieve relevant context from the FAISS vector store based on the query.

    Args:
        query: The search query to find relevant information.
        k: Number of relevant documents to retrieve.
        
    Returns:
        str: Formatted string of retrieved context.
    """
    if not os.path.exists(FAISS_INDEX_PATH):
        return "Error: FAISS index not found. Please run `process_and_store_pdfs` first."

    try:
        # Added allow_dangerous_deserialization=True
        vectorstore = FAISS.load_local(FAISS_INDEX_PATH, embedding_model, allow_dangerous_deserialization=True)
        # No need to call get_embedding separately, similarity_search handles it
        results: list[Document] = vectorstore.similarity_search(query, k=k)

        if not results:
            return "No relevant context found for the query."

        context_str = "\\n---\\n".join([doc.page_content for doc in results])
        return f"Retrieved context:\\n{context_str}"

    except Exception as e:
        return f"Error retrieving context from FAISS: {e}"

#Agents
graph_agent = create_react_agent(
    model=get_llm_graph(), # Call factory function
    tools=[plot_line, plot_histogram],
    name="graph_expert",
    prompt="""/no_think /nothinking
    You are a graph expert. Always use one tool at a time.
    If you are given only the file name, for example, data.csv, look for the file in the following directory src\data"""
)

rag_agent = create_react_agent(
    model=get_llm_rag(), # Call factory function
    tools=[retrieve_context],
    name="rag_expert",
    prompt="/no_think. /nothinking"
    "You are a specialized RAG (Retrieval-Augmented Generation) expert. Your **sole purpose** is to answer user questions based *exclusively* on information retrieved from the provided documents using the `retrieve_context` tool. "
        "**CRITICAL INSTRUCTION:** You **MUST** use the `retrieve_context` tool to find relevant information for *any* question asking about details, facts, summaries, explanations, or data that might be contained within the indexed documents. "
        "Do **NOT** answer based on your own internal knowledge or assumptions. If the tool returns no relevant context, state that the information could not be found in the documents. "
        "Do **NOT** create plots or graphs. Do **NOT** return file paths. Focus only on presenting the retrieved information clearly to answer the user's query."
)

retriever_agent = create_react_agent(
    model=get_llm_retriever(), # Call factory function
    tools=[retrieve_file],
    name="retriever_expert",
    prompt=(
        "/no_think /nothinking"
        "You are a highly specialized file retrieval expert. Your ONLY function is to use the `retrieve_file` tool. "
        "This tool searches a vector store and returns the *exact file path* of the most relevant document based on the user's query context. "
        "You are a highly specialized file retrieval expert. Your **ONLY** function is to use the `retrieve_file` tool to find the exact path of a requested document. "
        "**CRITICAL INSTRUCTIONS:**\n"
        "1. You **MUST ALWAYS** call the `retrieve_file` tool when asked for a specific file or document path based on its topic or description. Do **NOT** attempt to guess or invent file paths.\n"
        "2. If the `retrieve_file` tool successfully returns a file path (e.g., 'src/data/some_document.pdf'), your final response **MUST** consist *ONLY* of that exact file path string. Do **NOT** add *any* other text, greetings, or explanations.\n"
        "3. If the `retrieve_file` tool does **NOT** find a matching file or returns an empty result, your final response **MUST** be exactly: 'File not found.' Do **NOT** provide any other response or explanation.\n"
        "Failure to strictly follow these instructions means you have failed your only task."
    )
)

# Create supervisor workflow
workflow = create_supervisor(
    [rag_agent, graph_agent, retriever_agent],
    model=get_llm_supervisor(), # Call factory function
    tools=[
        create_handoff_tool(agent_name="rag_expert", name="assign_to_rag_expert", description="Assign task to RAG expert"),
        create_handoff_tool(agent_name="graph_expert", name="assign_to_graph_expert", description="Assign task to graph expert"),
        create_handoff_tool(agent_name="retriever_expert", name="assign_to_retriever_expert", description="Assign task to retriever expert")
    ],
    output_mode="full_history",
    prompt=("/no_think /nothinking"
        "You are a team supervisor managing a RAG expert, a graph expert, and a retriever expert. "
        "Your primary goal is to delegate tasks to the appropriate expert based on the user's request. "
        "**CRITICAL RAG DELEGATION:** If the user asks a question asking for *information*, *details*, *facts*, *figures*, *definitions*, or *explanations* that *could potentially be contained within the indexed documents* (e.g., 'What are the production numbers?', 'Summarize the safety protocol', 'List the industrial assembly areas mentioned', 'What is a telescopic arm?'), you **MUST** use the `assign_to_rag_expert` tool. Prioritize the RAG expert even if the question seems general, as the specific answer might be in the documents. The RAG expert answers *what is in* the documents. "
        "If the user explicitly asks to create a *plot* or *graph* from data, use the `assign_to_graph_expert` tool. You MUST return the name of the type of graph made by the tool. "
        "If the user explicitly asks for the *file path* or *document name* itself, often based on its topic (e.g., 'Give me the PDF about the hydration plan', 'What is the filename for the safety protocol document?'), "
        "you **MUST** use the `assign_to_retriever_expert` tool. The retriever expert ONLY provides the file path, NOT the content. You MUST submit a message of the form 'The document you are looking for is', FOLLOWED by the file path. You MUST provide the full path of the file to the user."
        "**Contrast:** Crucially, distinguish between asking *about content* (RAG) and asking *for the file itself* (Retriever). "
        "Only respond directly for purely conversational queries (e.g., 'Hello', 'Thank you') that do not require information retrieval, file path retrieval, or graph generation. "
        "Never try to answer content questions or find file paths yourself."
        "You MUST always respond in Spanish."
    )
)

# Compile and run
# Compile with checkpointer/store
app = workflow.compile(
    checkpointer=checkpointer,
    store=store
)

if __name__ == "__main__":
    # IMPORTANT: You need to call process_and_store_pdfs before running the workflow
    # Example: process_and_store_pdfs(["path/to/your/doc1.pdf", "path/to/your/doc2.pdf"])
    # --- PDF Processing ---
    pdf_files_to_load = ["src\data\mineria_chilena_removed.pdf", 
                        'src\data\Mina_Ministro_Hales.pdf', 
                        'src\data\Manual_curso_de_maquinaria_pesada-Retroexcavadora.pdf',
                        r'src\data\2023_03_31_codelco_entrega_resultados_4t_2022.pdf']
    # Check if index needs processing or skip if exists (optional optimization)
    # Force processing now to apply new chunking
    # if not os.path.exists(FAISS_INDEX_PATH):
    #      print(f"FAISS index not found at {FAISS_INDEX_PATH}. Processing PDFs...")
    #      process_and_store_pdfs(pdf_files_to_load)
    # else:
    #      print(f"FAISS index found at {FAISS_INDEX_PATH}. Skipping PDF processing.")
    print("Processing PDFs and rebuilding FAISS index...")
    process_and_store_pdfs(pdf_files_to_load)
    print("PDF processing and index rebuild complete.")


    # --- Interactive Conversation ---

    # Generate a unique conversation ID for this session
    session_id = uuid.uuid4()
    conversation_id = f"interactive-session-{session_id}"
    config = {"configurable": {"thread_id": conversation_id}}

    print(f"Starting new conversation with ID: {conversation_id}")
    print("Enter 'quit' to exit.")

    while True:
        user_input = input("\nYou: ")
        if user_input.lower() == 'quit':
            print("Ending conversation.")
            break

        print("\nAssistant:")
        # Invoke the app with the user input and the conversation config
        result = app.invoke({"messages": [("user", user_input)]}, config=config)

        # Print the full message history for this turn
        if result and "messages" in result:
            all_messages_this_turn = result["messages"]
            # Often the first message is the user input again, skip it if needed
            # or just print everything including the input.
            # Let's print from the second message onwards to see the flow clearly
            # Adjust the range if you want to see the user input repeated
            if len(all_messages_this_turn) > 1:
                 print("\n--- Turn History ---")
                 for message in all_messages_this_turn[1:]: # Start from index 1 to exclude user input repetition
                     message.pretty_print()
                 print("--- End Turn History ---")
            elif all_messages_this_turn:
                 # Handle case where there's only the user message (e.g., error)
                 print("Assistant: (No further messages generated in trace)")
                 all_messages_this_turn[0].pretty_print()
        else:
            print("Assistant: (No response generated or unexpected result format.)")


    # --- Optional: Display Final State/History ---

    # Display the final graph state (optional)
    # try:
    #     display(Image(app.get_graph(config=config).draw_mermaid_png()))
    # except Exception as e:
    #     print(f"Could not display graph: {e}")

    # Print the full conversation history for the thread (optional)
    # print("\n--- Full Conversation History ---")
    # try:
    #     full_history = app.get_state(config)
    #     if full_history and 'messages' in full_history.values:
    #         for m in full_history.values['messages']:
    #             m.pretty_print()
    #     else:
    #         print("Could not retrieve conversation history.")
    # except Exception as e:
    #      print(f"Error retrieving history: {e}")
