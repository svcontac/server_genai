import asyncio
import json
import ragas.messages as r
import os # Import os module

# Funciones para calcular las metricas
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import AgentGoalAccuracyWithReference, ToolCallAccuracy

# Variables importadas
from models import get_llm_evaluator, MODEL_NAME_RETRIEVER, EVALUATION_MODEL_NAME
from metrics_calculator import main

if __name__=="__main__":
    agent = 'recuperador'
    RESULTS_FILE_PATH = r"metricas\recuperador.json" # Define path

    # Try to load existing results, initialize if not found
    try:
        with open(RESULTS_FILE_PATH, "r") as f:
            result = json.load(f)
    except FileNotFoundError:
        print(f"Results file not found at {RESULTS_FILE_PATH}. Initializing fresh results.")
        result = {agent: {}} # Initialize basic structure
    except json.JSONDecodeError:
        print(f"Error decoding JSON from {RESULTS_FILE_PATH}. Initializing fresh results.")
        result = {agent: {}} # Initialize basic structure

    # Ensure the agent key exists and add the model key using the RETRIEVER model name
    if agent not in result:
        result[agent] = {}
    result[agent].update({MODEL_NAME_RETRIEVER:{}})

    # Funciones para calcular la métricas
    goal_scorer = AgentGoalAccuracyWithReference()
    evaluator_llm_instance = get_llm_evaluator()
    evaluator_llm_wrapper = LangchainLLMWrapper(evaluator_llm_instance)
    goal_scorer.llm = evaluator_llm_wrapper
    tool_scorer = ToolCallAccuracy()

    #Diccionario de metricas
    scorer_dict = {'goal': goal_scorer, 'tool': tool_scorer}

    # 1. Define sample user inputs and expected tool calls
    test_cases = [
        {
            "query": r"Puedes entregarme el archivo pdf que hable sobre el plan de humectación",
            "expected_tool_calls": [
                r.ToolCall(name='assign_to_retriever_expert', args={}),
                r.ToolCall(name='retrieve_file', args={'query': 'plan de humectación'}),
                r.ToolCall(name='transfer_back_to_supervisor', args={}),
            ],
            "description": "Test Case 1: Plan de Humectación",
            "reference": 'El documento que estás buscando es src\data\Mina_Ministro_Hales.pdf',
            'id': 0
        },
        {
            "query": r"Puedes entregarme el archivo pdf que hable sobre la estrategia nacional del Litio",
            "expected_tool_calls": [
                r.ToolCall(name='assign_to_retriever_expert', args={}),
                r.ToolCall(name='retrieve_file', args={'query': 'estrategia nacional del Litio'}),
                r.ToolCall(name='transfer_back_to_supervisor', args={}),
            ],
            "description": "Test Case 2: Estrategia Nacional del Litio",
            "reference": 'El documento que estás buscando es src\data\mineria_chilena_removed.pdf',
            'id': 1
        },
        {
            "query": r"Puedes entregarme el archivo pdf que explique de forma detallada cuál la diferencia entre una excavadora y la retroexcavadora.",
            "expected_tool_calls": [
                r.ToolCall(name='assign_to_retriever_expert', args={}),
                r.ToolCall(name='retrieve_file', args={'query': 'diferencia entre excavadora y la retroexcavadora'}),
                r.ToolCall(name='transfer_back_to_supervisor', args={}),
            ],
            "description": "Test Case 3: Diferencia entre una excavadora y la retroexcavadora",
            "reference": 'El documento que estás buscando es src\data\Manual_curso_de_maquinaria_pesada-Retroexcavadora.pdf',
            'id': 2
        },
        {
            "query": r"Puedes entregarme el archivo pdf que mencione el proyecto de Chuquicamata Subterránea.",
            "expected_tool_calls": [
                r.ToolCall(name='assign_to_retriever_expert', args={}),
                r.ToolCall(name='retrieve_file', args={'query': 'proyecto de Chuquicamata Subterránea'}),
                r.ToolCall(name='transfer_back_to_supervisor', args={}),
            ],
            "description": "Test Case 4: Proyecto de Chuquicamata Subterránea",
            "reference": r'El documento que estás buscando es src\data\2023_03_31_codelco_entrega_resultados_4t_2022.pdf',
            'id': 3
        },
        {
            "query": r"Entregame el achivo que menciona el proyecto de la divisón DMH.",
            "expected_tool_calls": [
                r.ToolCall(name='assign_to_retriever_expert', args={}),
                r.ToolCall(name='retrieve_file', args={'query': 'proyecto de la divisón DMH'}),
                r.ToolCall(name='transfer_back_to_supervisor', args={}),
            ],
            "description": "Test Case 5: Proyecto de la divisón DMH",
            "reference": 'El documento que estás buscando es src\data\Mina_Ministro_Hales.pdf',
            'id': 4
        },
        {
            "query": r"¿Qué archivo menciona el significado de RMI?",
            "expected_tool_calls": [
                r.ToolCall(name='assign_to_retriever_expert', args={}),
                r.ToolCall(name='retrieve_file', args={'query': 'significado de RMI'}),
                r.ToolCall(name='transfer_back_to_supervisor', args={}),
            ],
            "description": "Test Case 6: Significado de RMI",
            "reference": 'El documento que estás buscando es src\data\mineria_chilena_removed.pdf',
            'id': 5
        },
        {
            "query": r"Retorna algún archivo que menciones las partes de un sistema de frenos.",
            "expected_tool_calls": [
                r.ToolCall(name='assign_to_retriever_expert', args={}),
                r.ToolCall(name='retrieve_file', args={'query': 'partes de un sistema de frenos'}),
                r.ToolCall(name='transfer_back_to_supervisor', args={}),
            ],
            "description": "Test Case 7: Partes de un sistema de frenos",
            "reference": 'El documento que estás buscando es src\data\Manual_curso_de_maquinaria_pesada-Retroexcavadora.pdf',
            'id': 6
        },
        {
            "query": r"Necesito un documento que me entrege la información númerica de la ganancia bruta del cobre en el año 2022.",
            "expected_tool_calls": [
                r.ToolCall(name='assign_to_retriever_expert', args={}),
                r.ToolCall(name='retrieve_file', args={'query': 'ganancia bruta del cobre en el año 2022'}),
                r.ToolCall(name='transfer_back_to_supervisor', args={}),
            ],
            "description": "Test Case 8: Ganancia bruta del cobre",
            "reference": r'El documento que estás buscando es src\data\2023_03_31_codelco_entrega_resultados_4t_2022.pdf',
            'id': 7
        },
    ]

    # Run the async main function - Pass the RETRIEVER model name
    result = asyncio.run(main(test_cases, result, agent, MODEL_NAME_RETRIEVER, scorer_dict))
    print('-- Calculo de métricas completado exitosamente -- ')

    # Ensure the directory exists before saving
    os.makedirs(os.path.dirname(RESULTS_FILE_PATH), exist_ok=True)

    with open(RESULTS_FILE_PATH, "w") as f:
        json.dump(result, f, indent=3)
        print(f'-- Métricas guardadas en el archivo {RESULTS_FILE_PATH} --') # Use path variable