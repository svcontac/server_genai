import json
import time
import uuid
import logging # Import logging
import os # Import os

# Funciones para calcular las metricas
from copy import deepcopy
from langchain_core.messages import AIMessage, HumanMessage
from ragas.dataset_schema import MultiTurnSample, SingleTurnSample
from ragas.integrations.langgraph import convert_to_ragas_messages
# Base Metric class and specific metric types for checks
from ragas.metrics import (
    Metric,
    ToolCallAccuracy,
    AgentGoalAccuracyWithReference # Add other multi-turn metrics if needed
    # Import single-turn metrics only if needed for explicit checks, otherwise rely on 'else'
    # AnswerAccuracy, BleuScore, RougeScore, NonLLMStringSimilarity
)
# MetricType might not be needed if we check instance types
# from ragas.metrics import MetricType

# Variables importadas
from flujo_v2 import app

# Function to process messages for Ragas format
def process_messages_for_ragas(original_messages):
    processed_messages = []
    for msg in original_messages:
        if isinstance(msg, AIMessage) and getattr(msg, 'tool_calls', None):
            new_msg = deepcopy(msg)
            new_additional_kwargs = deepcopy(new_msg.additional_kwargs) if new_msg.additional_kwargs else {}
            ragas_formatted_tool_calls = []
            for tool_call in new_msg.tool_calls:
                call_name = tool_call.get('name') if isinstance(tool_call, dict) else getattr(tool_call, 'name', None)
                call_args = tool_call.get('args') if isinstance(tool_call, dict) else getattr(tool_call, 'args', None)
                call_id = tool_call.get('id') if isinstance(tool_call, dict) else getattr(tool_call, 'id', None)

                if call_name and call_args is not None:
                    try:
                        arguments_json = json.dumps(call_args)
                        formatted_call = {
                            "function": {
                                "name": call_name,
                                "arguments": arguments_json,
                            },
                            "id": call_id
                        }
                        ragas_formatted_tool_calls.append(formatted_call)
                    except TypeError as e:
                        print(f"Warning: Could not JSON-serialize args for tool call {call_name}: {e}")

            if ragas_formatted_tool_calls:
                new_additional_kwargs["tool_calls"] = ragas_formatted_tool_calls
                processed_messages.append(AIMessage(
                    content=new_msg.content,
                    additional_kwargs=new_additional_kwargs,
                    response_metadata=new_msg.response_metadata,
                    id=new_msg.id,
                    name=new_msg.name,
                    tool_call_chunks=getattr(new_msg, 'tool_call_chunks', None)
                ))
            else:
                processed_messages.append(msg)
        else:
            processed_messages.append(msg)
    return processed_messages



# Async function to run a single test case
async def run_test_case(test_case_data, scorer_dict, agent, model_name):
    test_case_id = test_case_data["id"] # Assume ID exists
    description = test_case_data["description"]

    # Sanitize model_name for use in directory paths
    sanitized_model_name = model_name.replace("/", "_").replace(":", "_")

    # Construct the new log path using sanitized name
    log_dir = os.path.join("logs", agent, sanitized_model_name)
    log_filename = os.path.join(log_dir, f"{agent}_tc_{test_case_id}.log")

    # Ensure the full directory path exists before logging
    os.makedirs(log_dir, exist_ok=True)

    # --- Logger Setup --- 
    logger = logging.getLogger(f"test_case_{agent}_{model_name}_{test_case_id}") # Make logger name more unique
    logger.setLevel(logging.INFO)
    logger.propagate = False # Prevent duplicate logging if root logger is configured

    # Remove existing handlers from previous runs
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
        handler.close()

    # File Handler (mode='w' to overwrite for each run)
    file_handler = logging.FileHandler(log_filename, mode='w')
    file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    # Console Handler
    stream_handler = logging.StreamHandler()
    stream_formatter = logging.Formatter('%(levelname)s: %(message)s') # Simpler format for console
    stream_handler.setFormatter(stream_formatter)
    logger.addHandler(stream_handler)
    # --- End Logger Setup ---

    sample_user_query = test_case_data["query"]
    reference = test_case_data.get('reference')
    reference_tool_calls = test_case_data.get("expected_tool_calls", [])

    logger.info(f"--- Running {description} (Log file: {log_filename}) ---")
    logger.info(f"User Query: {sample_user_query}")
    if reference:
        logger.info(f"Reference: {reference}")
    if reference_tool_calls:
        logger.info(f"Expected Tool Calls: {[f'{tc.name}({tc.args})' for tc in reference_tool_calls]}")

    test_session_id = uuid.uuid4()
    test_conversation_id = f"test-run-{test_session_id}"
    test_config = {"configurable": {"thread_id": test_conversation_id}}

    logger.info(f"Running test with conversation ID: {test_conversation_id}")

    input_data = {"messages": [HumanMessage(content=sample_user_query)]}

    start_time = time.time()
    try:
        result = app.invoke(input_data, config=test_config)
    except Exception as e:
        logger.error(f"Error during app.invoke: {e}", exc_info=True) # Log exception info
        result = {} # Ensure result is a dict
    end_time = time.time()
    inference_time = end_time - start_time

    metrics = {'time': inference_time}

    logger.info("--- Workflow Output Messages ---")
    generated_answer = None
    ragas_trace = None
    all_messages = result.get("messages", [])

    if all_messages:
        for message in all_messages:
            # Log the detailed representation of each message
            try:
                logger.info(message.pretty_repr())
            except Exception as log_e:
                logger.error(f"Error logging message representation: {log_e}")

        last_message = all_messages[-1]
        if isinstance(last_message, AIMessage):
            generated_answer = last_message.content if last_message.content is not None else ""

        try:
            processed_messages = process_messages_for_ragas(all_messages)
            ragas_trace = convert_to_ragas_messages(processed_messages)
            logger.info("--- Ragas Trace Conversion Successful ---")
        except Exception as e:
            logger.error(f"Error converting messages to Ragas trace: {e}")
            ragas_trace = [] # Set to empty list on error

    else:
        logger.warning("Warning: No messages found in the result.")

    for name, scorer in scorer_dict.items():
        score = None
        try:
            if not isinstance(scorer, Metric):
                logger.warning(f"Warning: Scorer '{name}' is not a valid Ragas Metric instance. Skipping.")
                continue

            logger.info(f"--- Calculating {name} score for {description} ---")

            if isinstance(scorer, (ToolCallAccuracy, AgentGoalAccuracyWithReference)):
                if ragas_trace is None:
                    logger.warning(f"Skipping multi-turn metric '{name}': Ragas trace unavailable.")
                    continue

                # Determine the trace to use based on agent and metric
                current_trace_for_scoring = ragas_trace
                if isinstance(scorer, ToolCallAccuracy) and agent == 'supervisor':
                    if len(ragas_trace) >= 2:
                        # Attempt to find the first Human and first AI message
                        first_human_msg = ragas_trace[0] if ragas_trace[0].type == 'human' else None
                        first_ai_msg = next((msg for msg in ragas_trace[1:] if msg.type == 'ai'), None)

                        if first_human_msg and first_ai_msg:
                            truncated_trace = [first_human_msg, first_ai_msg]
                            logger.info("--- Using TRUNCATED Ragas Trace for Supervisor ToolCallAccuracy ---")
                            current_trace_for_scoring = truncated_trace
                        else:
                            logger.warning("Warning: Could not create truncated trace (missing Human or first AI msg). Using full trace for Supervisor ToolCallAccuracy.")
                    else:
                         logger.warning("Warning: Trace too short for truncation. Using full trace for Supervisor ToolCallAccuracy.")
                # For all other cases (other metrics, other agents), current_trace_for_scoring remains the full ragas_trace

                # Prepare sample args using the determined trace
                sample_args = {'user_input': current_trace_for_scoring}
                if isinstance(scorer, ToolCallAccuracy):
                    sample_args['reference_tool_calls'] = reference_tool_calls
                elif isinstance(scorer, AgentGoalAccuracyWithReference):
                    if reference is None:
                        logger.warning(f"Skipping multi-turn metric '{name}': Reference goal not provided.")
                        continue
                    sample_args['reference'] = reference
                else:
                    if reference is not None:
                        sample_args['reference'] = reference
                    else:
                        logger.warning(f"Warning: No specific handling for multi-turn metric '{name}' and no 'reference' found. Scoring may fail.")

                sample = MultiTurnSample(**sample_args)
                score = await scorer.multi_turn_ascore(sample)

            else:
                if generated_answer is None:
                    logger.warning(f"Skipping single-turn metric '{name}': Generated answer unavailable.")
                    continue
                if reference is None:
                    logger.warning(f"Skipping single-turn metric '{name}': Reference answer not provided.")
                    continue

                sample = SingleTurnSample(
                    user_input=sample_user_query,
                    reference=reference,
                    response=generated_answer,
                )
                score = await scorer.single_turn_ascore(sample)

            if score is not None:
                logger.info(f'{name}: {score:.4f}')
                metrics[name] = score
            else:
                logger.info(f"{name}: Calculation failed or skipped.")

        except AttributeError as ae:
            logger.error(f"Error calculating score for metric '{name}': Missing expected method ({ae}). Check metric type and Ragas version.")
        except Exception as e:
            logger.error(f"Error calculating score for metric '{name}': {e}", exc_info=True) # Log exception info

    logger.info(f"--- End of {description} ---")

    # --- Logger Cleanup ---
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
        handler.close()
    # --- End Logger Cleanup ---

    return metrics

async def main(test_cases, result_dict, agent, model_name, scorer_dict):
    # Ensure base logs directory exists (nested dirs handled in run_test_case)
    # os.makedirs("logs", exist_ok=True) # This is now handled within run_test_case

    if agent not in result_dict:
        result_dict[agent] = {}
    if model_name not in result_dict[agent]:
        result_dict[agent][model_name] = {}

    for test_case in test_cases:
        if 'id' not in test_case:
            # Use basic logging if logger setup failed previously
            logging.warning(f"Warning: Test case '{test_case.get('description', 'N/A')}' missing 'id'. Skipping.")
            continue

        test_case_id = test_case['id']
        # Pass model_name to run_test_case
        metrics = await run_test_case(test_case, scorer_dict, agent, model_name)

        if metrics:
            if test_case_id not in result_dict[agent][model_name]:
                result_dict[agent][model_name][test_case_id] = {}
            result_dict[agent][model_name][test_case_id]['metricas'] = metrics
            result_dict[agent][model_name][test_case_id]['description'] = test_case.get('description')
        else:
            # Construct the expected log path for the warning message using sanitized name
            sanitized_model_name_for_warning = model_name.replace("/", "_").replace(":", "_")
            expected_log_path = os.path.join("logs", agent, sanitized_model_name_for_warning, f"{agent}_tc_{test_case_id}.log")
            logging.getLogger().warning(f"No metrics calculated for test case ID {test_case_id}. Check log: {expected_log_path} for details.")

    return result_dict