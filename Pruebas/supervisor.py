import asyncio
import json
import ragas.messages as r

# Funciones para calcular las metricas
from ragas.llms import LangchainLLMWrapper
# Only import ToolCallAccuracy
from ragas.metrics import ToolCallAccuracy

# Variables importadas
from models import MODEL_NAME_SUPERVISOR
from metrics_calculator import main

if __name__=="__main__":
    agent = 'supervisor'  # Evaluate the supervisor agent
    # Se carga el diccionario para guardar los resultados
    # Use a dedicated file for supervisor metrics
    try:
        with open(r"metricas\supervisor.json", "r") as f:
            result = json.load(f)
    except FileNotFoundError:
        # Create the base structure if the file doesn't exist
        result = {agent: {}}
    result[agent].update({MODEL_NAME_SUPERVISOR:{}})


    # Funciones para calcular la métricas
    # Remove AgentGoalAccuracyWithReference scorer
    # goal_scorer = AgentGoalAccuracyWithReference()
    # evaluator_llm = LangchainLLMWrapper(model)
    # goal_scorer.llm = evaluator_llm
    tool_scorer = ToolCallAccuracy()

    #Diccionario de metricas - Only include tool_scorer
    scorer_dict = {'tool': tool_scorer}

    # 1. Define sample user inputs and expected tool calls for the supervisor
    #    The expected tool call is just the first delegation action.
    #    The reference describes the goal of correct delegation.
    test_cases = [
        # Test cases derived from recuperador_d.py
        {
            "query": r"Puedes entregarme el archivo pdf que hable sobre el plan de humectación",
            "expected_tool_calls": [
                r.ToolCall(name='assign_to_retriever_expert', args={}),
            ],
            "description": "TC 1 (Recuperador): Delegate Humectacion Query",
            "reference": 'The goal is to assign the task to the retriever expert.',
            'id': 0
        },
        {
            "query": r"Puedes entregarme el archivo pdf que hable sobre la estrategia nacional del Litio",
            "expected_tool_calls": [
                r.ToolCall(name='assign_to_retriever_expert', args={}),
            ],
            "description": "TC 2 (Recuperador): Delegate Litio Query",
            "reference": 'The goal is to assign the task to the retriever expert.',
            'id': 1
        },
        # Test cases derived from graficador_d.py
        {
            "query":  r"Genera un gráfico de línea de la primera columna vs la segunda columna del archivo data.csv",
            "expected_tool_calls": [
                r.ToolCall(name='assign_to_graph_expert', args={'plot_type': 'line'}),
            ],
            "description": "TC 3 (Graficador): Delegate Line Plot Query",
            "reference": 'The goal is to assign the task to the graph expert.',
            'id': 2
        },
        {
            "query":  "Genera un histograma de la columna 1 del archivo data.csv",
            "expected_tool_calls": [
                r.ToolCall(name='assign_to_graph_expert', args={'plot_type': 'histogram'}),
            ],
            "description": "TC 4 (Graficador): Delegate Histogram Query",
            "reference": 'The goal is to assign the task to the graph expert.',
            'id': 3
        },
        # Test cases derived from metricas_rag.py
        {
            "query": "¿Cuál es la producción diaria estimada para la mina Ministro Hales?",
            "expected_tool_calls": [
                r.ToolCall(name='assign_to_rag_expert', args={})
            ],
            "description": "TC 5 (RAG): Delegate Producción Ministro Hales Query",
            "reference": "The goal is to assign the task to the RAG expert.",
            'id': 4
        },
        {
            "query": "¿Cuáles son las áreas de montaje industrial del proyecto de la mina Ministro Hales? Menciónalas",
            "expected_tool_calls": [
                r.ToolCall(name='assign_to_rag_expert', args={})
            ],
            "description": "TC 6 (RAG): Delegate Áreas Montaje Query",
            "reference": "The goal is to assign the task to the RAG expert.",
            'id': 5
        },
        {
            "query": "¿Cuál es la inversión que contempló el proyecto de la mina Ministro Hales?",
            "expected_tool_calls": [
                r.ToolCall(name='assign_to_rag_expert', args={})
            ],
            "description": "TC 7 (RAG): Delegate Inversión Proyecto Query",
            "reference": "The goal is to assign the task to the RAG expert.",
            'id': 6
        },
        {
            "query": "¿Cuál es el mineral más importante en la producción minera de Chile?",
            "expected_tool_calls": [
                r.ToolCall(name='assign_to_rag_expert', args={})
            ],
            "description": "TC 8 (RAG): Delegate Mineral Importante Query",
            "reference": "The goal is to assign the task to the RAG expert.",
            'id': 7
        },
        {
            "query": "¿Cuál fue el porcentaje del PIB nacional correspondiente al cobre en el año 2023?",
            "expected_tool_calls": [
                r.ToolCall(name='assign_to_rag_expert', args={})
            ],
            "description": "TC 9 (RAG): Delegate PIB Cobre 2023 Query",
            "reference": "The goal is to assign the task to the RAG expert.",
            'id': 8
        },
        {
            "query": "¿Cuál fue la reserva de plata en el Anuario de la Minería de Chile del 2016?",
            "expected_tool_calls": [
                r.ToolCall(name='assign_to_rag_expert', args={})
            ],
            "description": "TC 10 (RAG): Delegate Reserva Plata 2016 Query",
            "reference": "The goal is to assign the task to the RAG expert.",
            'id': 9
        },
        {
            "query": "¿Puedes entregarme los componentes externos de una retroexcavadora?",
            "expected_tool_calls": [
                r.ToolCall(name='assign_to_rag_expert', args={})
            ],
            "description": "TC 11 (RAG): Delegate Componentes Retroexcavadora Query",
            "reference": "The goal is to assign the task to the RAG expert.",
            'id': 10
        },
        {
            "query": "¿Qué es un brazo telescópico para la retroexcavadora?",
            "expected_tool_calls": [
                r.ToolCall(name='assign_to_rag_expert', args={})
            ],
            "description": "TC 12 (RAG): Delegate Brazo Telescópico Query",
            "reference": "The goal is to assign the task to the RAG expert.",
            'id': 11
        }
    ]

    # Run the async main function - Pass the SUPERVISOR model name for results keying
    result = asyncio.run(main(test_cases, result, agent, MODEL_NAME_SUPERVISOR, scorer_dict))
    print('-- Calculo de métricas completado exitosamente -- ')

    # Save results to the supervisor's JSON file
    with open(r"metricas\supervisor.json", "w") as f:
        json.dump(result, f, indent=3)
        print('-- Métricas guardadas en el archivo supervisor.json --')