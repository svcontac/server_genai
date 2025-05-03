import asyncio
import json
import ragas.messages as r
import os # Import os module

# Funciones para calcular las metricas
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import AgentGoalAccuracyWithReference, ToolCallAccuracy

# Variables importadas
from models import get_llm_evaluator, MODEL_NAME_GRAPH, EVALUATION_MODEL_NAME
from metrics_calculator import main


if __name__=="__main__":
    agent = 'graficador'
    RESULTS_FILE_PATH = r"metricas\graficador.json" # Define path

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

    # Ensure the agent key exists and add the model key using the GRAPH model name
    if agent not in result:
        result[agent] = {}
    result[agent].update({MODEL_NAME_GRAPH:{}})

    # Funciones para calcular la métricas
    goal_scorer = AgentGoalAccuracyWithReference()
    evaluator_llm_instance = get_llm_evaluator()
    evaluator_llm_wrapper = LangchainLLMWrapper(evaluator_llm_instance)
    goal_scorer.llm = evaluator_llm_wrapper
    tool_scorer = ToolCallAccuracy()
    scorer_dict = {'goal': goal_scorer, 'tool': tool_scorer}

    # 1. Define sample user inputs and expected tool calls
    test_cases = [
        {
            "query":  "Genera un gráfico de línea de la primera columna vs la segunda columna del archivo data.csv",
            "expected_tool_calls": [
                r.ToolCall(name='assign_to_graph_expert', args={'plot_type': 'line'}),
                r.ToolCall(name='plot_line', args={'file_path': 'src/data/data.csv', 'x_col': 0, 'y_col': 1}),
                r.ToolCall(name='transfer_back_to_supervisor', args={}), 
            ],
            "description": "Test Case 1: Line Plot",
            "reference": 'Gráfico de línea (plot_line) realizado exitosamente.',
            'id': 0
        },
        {
            "query":  "Genera un histograma de la columna 1 del archivo data.csv",
            "expected_tool_calls": [
                r.ToolCall(name='assign_to_graph_expert', args={'plot_type': 'histogram'}),
                r.ToolCall(name='plot_histogram', args={'column': 1, 'file_path': 'src/data/data.csv'}),
                r.ToolCall(name='transfer_back_to_supervisor', args={}),
            ],
            "description": "Test Case 2: Histogram",
            "reference": 'Histograma (plot_histogram) realizado exitosamente.',
            'id': 1
        },
        {
            "query":  "Puedes graficar la primera variable a tráves del tiempo del data.csv",
            "expected_tool_calls": [
                r.ToolCall(name='assign_to_graph_expert', args={'plot_type': 'line'}),
                r.ToolCall(name='plot_line', args={'file_path': 'src/data/data.csv', 'x_col': 0, 'y_col': 'tiempo'}),
                r.ToolCall(name='transfer_back_to_supervisor', args={}), 
            ],
            "description": "Test Case 3: Line Plot - 2",
            "reference": 'Gráfico de línea (plot_line) realizado exitosamente.',
            'id': 2
        },
        {
            "query":  "Puedes graficar la frecuencia de la variable_1 de los datos en data.csv",
            "expected_tool_calls": [
                r.ToolCall(name='assign_to_graph_expert', args={'plot_type': 'histogram'}),
                r.ToolCall(name='plot_histogram', args={'column': "variable_1", 'file_path': 'src/data/data.csv'}),
                r.ToolCall(name='transfer_back_to_supervisor', args={}),
            ],
            "description": "Test Case 4: Histogram - 2",
            "reference": 'Histograma (plot_histogram) realizado exitosamente.',
            'id': 3
        },

    ]

    # Run the async main function - Pass the GRAPH model name
    result = asyncio.run(main(test_cases, result, agent, MODEL_NAME_GRAPH, scorer_dict))
    print('-- Calculo de métricas completado exitosamente -- ')

    # Ensure the directory exists before saving
    os.makedirs(os.path.dirname(RESULTS_FILE_PATH), exist_ok=True)

    with open(RESULTS_FILE_PATH, "w") as f:
        json.dump(result, f, indent=3)
        print(f'-- Métricas guardadas en el archivo {RESULTS_FILE_PATH} --') # Use path variable