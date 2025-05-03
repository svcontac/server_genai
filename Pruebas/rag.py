import asyncio
import json
import os
import ragas.messages as r

# Ragas metric imports
from ragas.metrics import (
    AnswerAccuracy,
    BleuScore,
    RougeScore,
    ToolCallAccuracy,
    NonLLMStringSimilarity
)
from ragas.llms import LangchainLLMWrapper

# Functions and variables from other modules
from models import get_llm_evaluator, MODEL_NAME_RAG, EVALUATION_MODEL_NAME # Import factory function
# Import the main runner function from metrics_calculator_d
from metrics_calculator import main

# --- Configuration ---
AGENT_NAME = 'rag'
# MODEL_NAME is imported from flujo_v2
RESULTS_FILE_PATH = r"metricas\rag.json" # Use raw string for Windows paths

# --- Test Cases ---
# Format matches the one expected by metrics_calculator_d.main
test_cases = [
    {
        "query": "¿Cuál es la producción diaria estimada para la mina Ministro Hales?",
        "reference": "Tendría una producción diaria de 50.000 toneladas de mineral.",
        "description": "RAG Test 1: Producción Ministro Hales",
        "expected_tool_calls": [
            r.ToolCall(name='assign_to_rag_expert', args={}),
            r.ToolCall(name='retrieve_context', args={'query': 'producción diaria estimada mina Ministro Hales'}),
            r.ToolCall(name='transfer_back_to_supervisor', args={})
        ],
        'id': 0
    },
    {
        "query": "¿Cuáles son las áreas de montaje industrial del proyecto de la mina Ministro Hales? Menciónalas",
        "reference": "áreas 3000, 5000 y 6000.",
        "description": "RAG Test 2: Áreas Montaje Industrial",
        "expected_tool_calls": [
            r.ToolCall(name='assign_to_rag_expert', args={}),
            r.ToolCall(name='retrieve_context', args={'query': 'áreas de montaje industrial proyecto mina Ministro Hales'}),
            r.ToolCall(name='transfer_back_to_supervisor', args={})
        ],
        'id': 1
    },
    {
        "query": "¿Cuál es la inversión que contempló el proyecto de la mina Ministro Hales?",
        "reference": "El proyecto tuvo una inversión superior a los US $ 3.000 millones",
        "description": "RAG Test 3: Inversión Proyecto",
        "expected_tool_calls": [
            r.ToolCall(name='assign_to_rag_expert', args={}),
            r.ToolCall(name='retrieve_context', args={'query': 'inversión proyecto mina Ministro Hales'}),
            r.ToolCall(name='transfer_back_to_supervisor', args={})
        ],
        'id': 2
    },
    {
        "query": "¿Cuál es el mineral más importante en la producción minera de Chile?",
        "reference": "El mineral más importante es el cobre.",
        "description": "RAG Test 4: Mineral más importante Chile",
        "expected_tool_calls": [
            r.ToolCall(name='assign_to_rag_expert', args={}),
            r.ToolCall(name='retrieve_context', args={'query': 'mineral más importante producción minera Chile'}),
            r.ToolCall(name='transfer_back_to_supervisor', args={})
        ],
        'id': 3
    },
    {
        "query": "¿Cuál fue el porcentaje del PIB nacional correspondiente al cobre en el año 2023?",
        "reference": "El porcentaje fue 73,1%",
        "description": "RAG Test 5: PIB Cobre 2023",
        "expected_tool_calls": [
            r.ToolCall(name='assign_to_rag_expert', args={}),
            r.ToolCall(name='retrieve_context', args={'query': 'porcentaje PIB nacional cobre 2023'}),
            r.ToolCall(name='transfer_back_to_supervisor', args={})
        ],
        'id': 4
    },
    {
        "query": "¿Cuál fue la reserva de plata en el Anuario de la Minería de Chile del 2016?",
        "reference": "La reserva fue de 25.542",
        "description": "RAG Test 6: Reserva Plata 2016",
        "expected_tool_calls": [
            r.ToolCall(name='assign_to_rag_expert', args={}),
            r.ToolCall(name='retrieve_context', args={'query': 'reserva de plata Anuario Minería Chile 2016'}),
            r.ToolCall(name='transfer_back_to_supervisor', args={})
        ],
        'id': 5
    },
     {
         "query": "¿Puedes entregarme los componentes externos de una retroexcavadora?",
         "reference": "Chasis o bastidor, cabina, ruedas u orugas, estabilizadores, brazo principal (boom), brazo secundario (dipper stick), cucharón (bucket), cargador frontal (front loader), sistemas hidráulicos, motor, sistema de escape, luces y espejos, placa de contrapeso.",
         "description": "RAG Test 7: Componentes Retroexcavadora",
         "expected_tool_calls": [
             r.ToolCall(name='assign_to_rag_expert', args={}),
             r.ToolCall(name='retrieve_context', args={'query': 'componentes externos retroexcavadora'}),
             r.ToolCall(name='transfer_back_to_supervisor', args={})
         ],
         'id': 6
     },
     {
         "query": "¿Qué es un brazo telescópico para la retroexcavadora?",
         "reference": "Un brazo telescópico es un tipo de brazo extensible que se puede alargar y acortar mediante un sistema de secciones deslizantes, similar a cómo funciona un telescopio.",
         "description": "RAG Test 8: Brazo Telescópico",
         "expected_tool_calls": [
             r.ToolCall(name='assign_to_rag_expert', args={}),
             r.ToolCall(name='retrieve_context', args={'query': 'brazo telescópico retroexcavadora'}),
             r.ToolCall(name='transfer_back_to_supervisor', args={})
         ],
         'id': 7
     },
     {
         "query": "¿Qué país estuve involucrado en la historia de la retroexcavadora?",
         "reference": "Los EE.UU.",
         "description": "RAG Test 9: Historia Retroexcavadora",
         "expected_tool_calls": [
             r.ToolCall(name='assign_to_rag_expert', args={}),
             r.ToolCall(name='retrieve_context', args={'query': 'país involucrado historia retroexcavadora'}),
             r.ToolCall(name='transfer_back_to_supervisor', args={})
         ],
         'id': 8
     },
    {
        "query": "¿Cuál fue la producción de toneladas de cobre fino de Codelco el año 2022?",
        "reference": "1.445.662 toneladas de cobre fino.",
        "description": "RAG Test 10: Producción Cobre Codelco 2022",
        "expected_tool_calls": [
            r.ToolCall(name='assign_to_rag_expert', args={}),
            r.ToolCall(name='retrieve_context', args={'query': 'producción toneladas cobre fino Codelco 2022'}),
            r.ToolCall(name='transfer_back_to_supervisor', args={})
        ],
        'id': 9
    },
    {
        "query": "¿Cuáles fueron las razones que explican los aumentos de los costos C1 en los resultados de Codelco del año 2022?",
        "reference": "Se explica principalmente por menos producción de cobre y subproductos y mayor precio de los insumos",
        "description": "RAG Test 11: Costos C1 Codelco 2022",
        "expected_tool_calls": [
            r.ToolCall(name='assign_to_rag_expert', args={}),
            r.ToolCall(name='retrieve_context', args={'query': 'razones aumento costos C1 resultados Codelco 2022'}),
            r.ToolCall(name='transfer_back_to_supervisor', args={})
        ],
        'id': 10
    },
    {
        "query": "¿Cuáles fueron los excedentes de los años 2022, 2021 y la variación respectiva (VAR) en los resultados de Codelco?",
        "reference": "El año 2022 excedentes de 2.769, el año 2021 de 7.478, se tiene una variación de –4.710",
        "description": "RAG Test 12: Excedentes Codelco 2021-2022",
        "expected_tool_calls": [
            r.ToolCall(name='assign_to_rag_expert', args={}),
            r.ToolCall(name='retrieve_context', args={'query': 'excedentes 2022 2021 variación resultados Codelco'}),
            r.ToolCall(name='transfer_back_to_supervisor', args={})
        ],
        'id': 11
    },
    {
        "query": "¿Cuál fue la ganancia bruta de Codelco del año 2022?",
        "reference": "4.734",
        "description": "RAG Test 13: Ganancia Bruta Codelco 2022",
        "expected_tool_calls": [
            r.ToolCall(name='assign_to_rag_expert', args={}),
            r.ToolCall(name='retrieve_context', args={'query': 'ganancia bruta Codelco 2022'}),
            r.ToolCall(name='transfer_back_to_supervisor', args={})
        ],
        'id': 12
    }
]

# --- Main Execution Logic ---
if __name__ == "__main__":

    # Load existing results or initialize
    if os.path.exists(RESULTS_FILE_PATH):
        try:
            with open(RESULTS_FILE_PATH, "r") as f:
                results = json.load(f)
        except json.JSONDecodeError:
            print(f"Warning: Could not decode JSON from {RESULTS_FILE_PATH}. Initializing fresh results.")
            results = {}
    else:
        print(f"Results file not found at {RESULTS_FILE_PATH}. Initializing fresh results.")
        results = {}

    # Ensure agent and model keys exist (using the RAG model name)
    if AGENT_NAME not in results:
        results[AGENT_NAME] = {}
    results[AGENT_NAME].update({MODEL_NAME_RAG:{}})

    # Instantiate Ragas Metrics required for this agent
    # Instantiate evaluator LLM when needed for the wrapper
    evaluator_llm_instance = get_llm_evaluator() # Call factory function
    evaluator_llm_wrapper = LangchainLLMWrapper(evaluator_llm_instance) # Use the instance
    answer_accuracy_scorer = AnswerAccuracy(llm=evaluator_llm_wrapper) # Use the wrapper
    bleu_scorer = BleuScore()
    rouge_scorer = RougeScore()
    non_llm_similarity_scorer = NonLLMStringSimilarity()
    tool_accuracy_scorer = ToolCallAccuracy()

    # Dictionary of scorers to pass to the main runner in metrics_calculator_d
    scorer_dict = {
        'answer_accuracy': answer_accuracy_scorer,
        'bleu': bleu_scorer,
        'rouge': rouge_scorer,
        'non_llm_similarity': non_llm_similarity_scorer,
        'tool': tool_accuracy_scorer # 'tool' key matches expected name in run_test_case
    }

    print(f"=== Running Evaluation for Agent: {AGENT_NAME}, Model: {MODEL_NAME_RAG} ===")

    # Check for FAISS index before running
    if not os.path.exists("faiss_index"):
         print("Error: FAISS index not found at 'faiss_index'.")
         print("Please run flujo_v2.py first to generate the index.")
         exit(1)
    else:
         print("FAISS index found.")

    # Run the evaluations using the imported main function from metrics_calculator_d
    try:
        print(f"Running {len(test_cases)} test cases...")
        # Pass results dict, agent name, model name (RAG), and the specific scorers for this agent
        updated_results = asyncio.run(main(test_cases, results, AGENT_NAME, MODEL_NAME_RAG, scorer_dict))
        print(f"--- Evaluation completed successfully ---")

        # Ensure the directory exists before saving
        os.makedirs(os.path.dirname(RESULTS_FILE_PATH), exist_ok=True)

        # Save the updated results
        with open(RESULTS_FILE_PATH, "w") as f:
            json.dump(updated_results, f, indent=4)
            print(f'-- Results saved to {RESULTS_FILE_PATH} --')

    except Exception as e:
        print(f"An error occurred during the evaluation run: {e}")
        print("Ensure the Ollama service is running and accessible.")
        import traceback
        traceback.print_exc()