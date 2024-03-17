
import argparse
import asyncio
import logging
import pandas as pd
import websockets
import aioconsole
from pathlib import Path
import numpy as np
import logging
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from scipy.stats import pearsonr, spearmanr
from nltk.translate.bleu_score import sentence_bleu
import json
import threading
from collections import deque


# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Global variables
user_data = {}
active_sessions = {"defender": {}, "attackers": []} 
prompt_queue = asyncio.Queue()
scores = {}
for i in range(1, 101):
    prompt_queue.put_nowait(f"Prompt #{i}")
    scores[f"Team_{i}"] = 0  # Initialize scores

def validate_file(file_path):
    if not Path(file_path).is_file():
        raise argparse.ArgumentTypeError(f"The file {file_path} does not exist.")
    return file_path

def load_dataset_templates(file_path):
    """
    Loads dataset templates from a JSON file, parsing each line as a JSON object.
    Stores configurations in `dataset_templates`, indexed by dataset name and config.
    This updated structure supports multiple answer fields, token join fields, and 
    a complex scoring data type that includes weights for different answers.
    """
    try:
        with open(file_path, 'r') as file:
            for line in file:
                try:
                    template = json.loads(line.strip())
                    key = f"{template.get('dataset', 'unknown_dataset')}_{template.get('config', 'default_config')}"

                    # Safe defaulting in case of missing fields in a template
                    dataset_templates[key] = {
                        "template": template.get("template", "No template provided."),
                        "answer_fields": template.get("answer_fields", []),
                        "answer_type": template.get("answer_type", "unknown"),
                        "evaluator_type": template.get("evaluator_type", "unknown"),
                        "token_join_fields": template.get("token_join_fields", {}),
                        "scoring_data": template.get("scoring_data", {})
                    }
                except json.JSONDecodeError as e:
                    logging.error(f"Error parsing JSON in line: {line}. Error: {e}")
                except Exception as e:
                    logging.error(f"Unexpected error processing line: {line}. Error: {e}")
    except FileNotFoundError:
        logging.error(f"The file {file_path} was not found.")
    except Exception as e:
        logging.error(f"Unexpected error opening file: {file_path}. Error: {e}")


# Global structure to store prompts and related information
unanswered_prompts = {}

# Example structure for an entry in `unanswered_prompts`
# "prompt_index": {
#     "prompt": "The actual prompt text",
#     "data": {...},  # Data related to the prompt
#     "template": {...},    # Scoring template
#     "attacks": [  # Supports multiple modifications (prefix, postfix, direct)
#         {
#             "attack_type": "prefix",
#             "modification": "The prefix modification"
#         },
#         {
#             "attack_type": "postfix",
#             "modification": "The postfix modification"
#         },
#         {
#             "attack_type": "direct",
#             "modification": "Modification to the main prompt"
#         }
#     ],
#     "attacker_team": "AttackerTeamName"
# }


# External mapping dictionary for evaluators
evaluators = {
    "exact_match": exact_match_evaluator,
    "numeric": numeric_evaluator,
    "f1_score": f1_score_evaluator,
    "bleu_score": bleu_score_evaluator,
    "accuracy": accuracy_evaluator,
    "precision": precision_evaluator,
    "recall": recall_evaluator,
    "pearson_correlation": pearson_correlation_evaluator,
    "spearman_correlation": spearman_correlation_evaluator,
}

async def evaluate_answer(evaluator_type, submitted_answer, correct_answers, scoring_data={}):
    """Dispatches the submitted answer to the appropriate evaluator based on the evaluator type."""
    evaluator_function = evaluators.get(evaluator_type)

    if evaluator_function is None:
        logging.error(f"Evaluator type '{evaluator_type}' not supported.")
        return 0

    try:
        if evaluator_type in ["accuracy", "precision", "recall", "numeric", "pearson_correlation", "spearman_correlation"]:
            return evaluator_function(submitted_answer, correct_answers, scoring_data)
        else:
            return evaluator_function(submitted_answer, correct_answers)
    except Exception as e:
        logging.error(f"Error during evaluation with {evaluator_type}: {str(e)}")
        return 0



def exact_match_evaluator(submitted_answer, correct_answers, scoring_data):
    return int(submitted_answer.strip() in [ans.strip() for ans in correct_answers])

def numeric_evaluator(submitted_answer, correct_answers, scoring_data):
    try:
        submitted_value = float(submitted_answer)
        correct_value = float(correct_answers[0])  # Assuming a single correct answer for simplicity
        tolerance = scoring_data.get('tolerance', 0)
        return int(abs(submitted_value - correct_value) <= tolerance)
    except (ValueError, TypeError):
        return 0

def f1_score_evaluator(submitted_answer, correct_answers, scoring_data):
    # Assuming the submitted answer and correct answers are tokenized into lists
    # This is a simplified example. Real usage might require adjustment.
    submitted_tokens = submitted_answer.split()
    correct_tokens = [ans.split() for ans in correct_answers]
    # Here we assume binary classification for the sake of simplicity
    binary_submitted = [1 if token in submitted_tokens else 0 for token in correct_tokens[0]]
    binary_correct = [1 for _ in correct_tokens[0]]
    return f1_score(binary_correct, binary_submitted, average='binary')

def bleu_score_evaluator(submitted_answer, correct_answers, scoring_data):
    submitted_tokens = submitted_answer.split()
    correct_tokens_lists = [[ans.split()] for ans in correct_answers]
    smoothing = SmoothingFunction().method1
    return sentence_bleu(correct_tokens_lists, submitted_tokens, smoothing_function=smoothing)

def accuracy_evaluator(submitted_answer, correct_answers, scoring_data):
    # Assuming binary classification; convert answers to numerical labels if needed
    submitted_label = int(submitted_answer)
    correct_labels = [int(ans) for ans in correct_answers]
    return accuracy_score(correct_labels, [submitted_label])

def precision_evaluator(submitted_answer, correct_answers, scoring_data):
    submitted_label = int(submitted_answer)
    correct_labels = [int(ans) for ans in correct_answers]
    return precision_score(correct_labels, [submitted_label], average='binary', zero_division=0)

def recall_evaluator(submitted_answer, correct_answers, scoring_data):
    submitted_label = int(submitted_answer)
    correct_labels = [int(ans) for ans in correct_answers]
    return recall_score(correct_labels, [submitted_label], average='binary', zero_division=0)

def pearson_correlation_evaluator(submitted_answer, correct_answers, scoring_data):
    submitted_values = np.array([float(submitted_answer)])
    correct_values = np.array([float(ans) for ans in correct_answers])
    correlation, _ = pearsonr(submitted_values, correct_values)
    return correlation

def spearman_correlation_evaluator(submitted_answer, correct_answers, scoring_data):
    submitted_values = np.array([float(submitted_answer)])
    correct_values = np.array([float(ans) for ans in correct_answers])
    correlation, _ = spearmanr(submitted_values, correct_values)
    return correlation


async def handle_submit_answer(websocket, index, submitted_answer, correct_andswers, scoring_data):
    # Ensure the function is called with the correct parameters:
    # `websocket` - the WebSocket connection of the defender
    # `index` - the index of the prompt being answered
    # `submitted_answer` - the answer provided by the defender

    # Check if the prompt exists and has not yet been answered
   
   try:
        if index not in unanswered_prompts:
            await websocket.send("ERROR: Prompt index not found or already answered.")
            return

        prompt_info = unanswered_prompts.get(index, {})
        evaluator_type = prompt_info.get('template', {}).get('evaluator_type', 'unknown')
        scoring_data = prompt_info.get('template', {}).get('scoring_data', {})
        answer_fields = prompt_info.get('template', {}).get('answer_fields', [])

        # Ensure correct_answers extraction logic handles missing data gracefully
        correct_answers = [prompt_info['data'].get(field, '') for field in answer_fields]

        if not correct_answers:
            logging.error(f"No correct answers found for prompt index {index}. Check 'answer_fields' in template.")
            await websocket.send("ERROR: No correct answers available for evaluation.")
            return

        score = await evaluate_answer(evaluator_type, submitted_answer, correct_answers, scoring_data)
        defender_feedback = f"Index: {index}, Answer received: '{submitted_answer}'. Score: {score}."

        await websocket.send(defender_feedback)

        # Feedback to attacker if present
        if 'attacks' in prompt_info:
            for attack in prompt_info['attacks']:
                attacker_team_name = attack.get('teamname')
                attacker_session = active_sessions["attackers"].get(attacker_team_name)
                if attacker_session:
                    attacker_websocket = attacker_session.get('websocket')
                    if attacker_websocket:
                        try:
                            await attacker_websocket.send(defender_feedback + " " + json.dumps(attack))
                        except Exception as e:
                            logging.error(f"Error sending feedback to attacker {attacker_team_name}: {str(e)}")

        logging.info(defender_feedback + " " + json.dumps(prompt_info))
        
        # Safe deletion of the prompt from the global structure
        unanswered_prompts.pop(index, None)

    except Exception as e:
        logging.error(f"Error handling submit answer for index {index}: {str(e)}")
        try:
            await websocket.send("ERROR: An internal error occurred while processing your submission.")
        except Exception as send_error:
            logging.error(f"Error sending error message to client: {str(send_error)}")


# Assuming direct_attacks_queue and attach_queue are accessed by multiple threads
direct_attacks_queue = deque()
attach_queue = {}
queue_lock = threading.Lock()

def generate_direct_attack_prompt():
    """Safely generates prompt data from the next direct attack in the queue."""
    with queue_lock:
        if not direct_attacks_queue:
            return None
        attack = direct_attacks_queue.popleft()
    return attack['id'], attack['prompt'], attack['data'], attack['template'], {}

def prompt_generator(file_path, chunk_size=1024):
    """Safely yields ID, prompt, template, and optional attack info from direct attacks or a file."""
    # Try to generate a prompt from the direct attacks queue
    direct_attack_prompt = generate_direct_attack_prompt()
    if direct_attack_prompt:
        yield direct_attack_prompt

    buffer = ''
    try:
        with open(file_path, 'r') as file:
            while True:
                chunk = file.read(chunk_size)
                if not chunk:
                    break
                buffer += chunk
                lines = buffer.split('\n')
                for line in lines[:-1]:
                    if line.startswith('Entry:'):
                        _, entry_json = line.split('Entry: ', 1)
                        entry = json.loads(entry_json.strip())
                        id, prompt, data, template = entry["idx"], entry["prompt"], entry.get("data", {}), entry["template"]

                        with queue_lock:
                            attach_mods = attach_queue.get(id, {})
                        prefix = attach_mods.get('prefix', '')
                        postfix = attach_mods.get('postfix', '')
                        modified_prompt = f"{prefix} {prompt} {postfix}".strip()

                        attacker_info = {'teamname': attach_mods.get('teamname', '')} if attach_mods else {}

                        yield id, modified_prompt, data, template, attacker_info
                buffer = lines[-1]
    except FileNotFoundError as e:
        logging.error(f"File not found: {e}")
    except json.JSONDecodeError as e:
        logging.error(f"Error parsing JSON: {e}")
    except Exception as e:
        logging.error(f"Unexpected error: {e}")

    # Check for any remaining content
    if buffer.startswith('Entry:'):
        _, entry_json = buffer.split('Entry: ', 1)
        entry = json.loads(entry_json.strip())
        yield entry["idx"], entry["prompt"], entry.get("data", {}), entry["template"], {}



async def handle_receive_prompt(websocket, session):
    try:
        prompt = next(prompts)  # Get the next prompt from the generator
        return prompt['prompt']
    except StopIteration:
        return "ERROR: No more prompts available."
    


def load_user_data(api_key_file):
    try:
        data = pd.read_csv(api_key_file)
        for _, row in data.iterrows():
            teamname = row['teamname']
            user_data[teamname] = {'apikey': row['apikey'], 'role': row['role']}
            scores[teamname] = 0  # Initialize score for each team
        logging.info("User data and initial scores loaded successfully.")
    except Exception as e:
        logging.error(f"Failed to load user data: {e}")
        exit(1)



async def authenticate(websocket):
    auth_message = await websocket.recv()
    parts = auth_message.strip().split()
    if len(parts) >= 3 and parts[0].lower() == "login":
        teamname, apikey = parts[1], parts[2]
        if teamname in user_data and user_data[teamname]['apikey'] == apikey:
            session_role = user_data[teamname]['role']
            # Defender stores a team name and websocket
            if session_role == "defender" and not active_sessions["defender"]:
                active_sessions["defender"] = {"teamname": teamname, "websocket": websocket}
            # Attackers stores a list of dictionaries, each with team name and websocket
            elif session_role == "attacker":
                active_sessions["attackers"].append({"teamname": teamname, "websocket": websocket})
            else:
                await websocket.send("A defender is already logged in.")
                return None
            return {"role": session_role, "teamname": teamname, "websocket": websocket}
        else:
            await websocket.send("Authentication failed. Please check your team name and API key.")
    return None
    
async def process_command(websocket, session, command):
    parts = command.strip().split()
    command_type = parts[0].lower()
    additional_parts = parts[1:] if len(parts) > 1 else []

    if command_type in ["login", "l"]:
        response = "Already logged in."
    elif command_type in ["logout", "lo"]:
        response = await handle_logout(websocket, session)
    elif command_type in ["score", "sc"]:
        response = display_scores(session)
    elif command_type in ["receive-prompt", "p"]:
        response = await handle_receive_prompt(websocket, session)
    elif command_type in ["submit-answer", "s"]:
        response = await handle_submit_answer(websocket, session, additional_parts)
    elif command_type in ["request-token", "r"]:
        response = await handle_request_token(websocket, session)
    elif command_type in ["submit-attack", "a"]:
        response = await handle_submit_attack(websocket, session, additional_parts)
    elif command_type in ["version", "v"]:
        response = "LLM Purple Test Competition CLI Version 1.0"
    elif command_type in ["help", "h"]:
        response = "Available commands: login(l), logout(lo), score(sc), receive-prompt(p), submit-answer(s), request-token(r), submit-attack(a), version(v), help(h)"
    else:
        response = "Unknown command or malformed input."
    
    await websocket.send(response)

async def handle_logout(websocket, session):
    if session["role"] == "defender":
        active_sessions["defender"] = None
    else:
        active_sessions["attackers"].remove(websocket)
    return "Logged out successfully."

def display_scores(websocket, session):
    score = scoreboard(websocket, session, 0)
    return f"{session['teamname']} score: {score}"

async def handle_receive_prompt(websocket, session, file_path):
    global unanswered_prompts, prompt_gen  # Assume prompt_gen is a global variable
    try:
        prompt_data = next(prompt_gen(file_path), None)  # Use the provided prompt generator
        if prompt_data:
            prompt_id, prompt_text, template = prompt_data['id'], prompt_data['prompt'], prompt_data['template']
            data = prompt_data.get('data', {})  # Extract data if available

            # Store the prompt details in unanswered_prompts for future reference
            unanswered_prompts[prompt_id] = {
                "prompt": prompt_text,
                "data": data,
                "template": template,
                "attacks": []  # Initialize an empty list for potential attacks
            }

            # Use the utility function to send the prompt text to the attacker
            await safe_websocket_send(websocket, prompt_text)
        else:
            await safe_websocket_send(websocket, "ERROR: No more prompts available.")
    except StopIteration:
        await safe_websocket_send(websocket, "ERROR: No more prompts available.")
    except Exception as e:
        logging.error(f"Error handling receive prompt request: {str(e)}")
        await safe_websocket_send(websocket, "ERROR: An internal error occurred while processing your request.")

async def handle_submit_answer(websocket, session, answer_parts):
    if session['role'] != "defender":
        return "ERROR: Only defenders can submit answers."
    answer = " ".join(answer_parts)
    result = submit_answer(websocket, session, answer)
    return f"Answer received: '{answer}'. Result: '{result}'. "

async def handle_request_token(websocket, session):
    if session['role'] != "attacker":
        return "ERROR: Only attackers can request tokens."
    token = request_attack_token(websocket, session)
    return "Token generated: '{token}'."


async def handle_submit_attack(websocket, session, parts):
    if session['role'] != "attacker":
        return "ERROR: Only attackers can submit attacks."
    attack_query, prefix, postfix, attach_to = None, None, None, None
    if "--prefix" in parts or "--postfix" in parts:
        if "--attach-to" not in parts:
            return "ERROR: --attach-to must be specified with --prefix or --postfix."
        for i, part in enumerate(parts):
            if part == "--prefix":
                prefix = parts[i + 1] if i + 1 < len(parts) else None
            elif part == "--postfix":
                postfix = parts[i + 1] if i + 1 < len(parts) else None
            elif part == "--attach-to":
                attach_to = parts[i + 1] if i + 1 < len(parts) else None
        if not prefix and not postfix:
            return "ERROR: Either --prefix or --postfix must be specified."
    else:
        if len(parts) == 1:
            attack_query = parts[0]
        else:
            return "ERROR: Invalid command syntax. Specify either an attack or use --prefix/--postfix with --attach-to."

    
    if attack_query:
        insert_direct_attack(websocket, session, attack_query)
        return f"Direct attack submitted: {attack_query}"
    else:
        insert_attack_queue(websocket, session, prefix, postfix, attach-to)
        return f"Attack submitted with prefix: '{prefix}', postfix: '{postfix}', attached to: {attach_to}."


async def handler(websocket, path):
    session = await authenticate(websocket)
    if session:
        try:
            async for message in websocket:
                await process_command(websocket, session, message)
        finally:
            if session and session["role"] == "defender":
                active_sessions["defender"] = None
            elif session and session["role"] == "attacker":
                active_sessions["attackers"].remove(websocket)
    else:
        await websocket.send("Authentication failed or not provided.")


async def console_input_handler():
    while True:
        command = await aioconsole.ainput(prompt=">")
        if command == "exit":
            logging.info("Exiting server...")
            # Attempt to close the defender session if it exists.
            if active_sessions["defender"]:
                defender_socket = active_sessions["defender"]["websocket"]
                await defender_socket.close()
            # Attempt to close all attacker sessions.
            for attacker in active_sessions["attackers"]:
                attacker_socket = attacker["websocket"]
                await attacker_socket.close()
            break
        elif command == "list":
            logging.info("Active sessions:")
            # Display the defender's session if it exists.
            if active_sessions["defender"]:
                logging.info(f"Defender: {active_sessions['defender']['teamname']}")
            else:
                logging.info("No active defender session.")
                
            # Display the attackers' sessions.
            if active_sessions["attackers"]:
                for attacker in active_sessions["attackers"]:
                    logging.info(f"Attacker: {attacker['teamname']}")
            else:
                logging.info("No active attacker sessions.")
        elif command.startswith("kick"):
            parts = command.split()
            if len(parts) == 2:
                teamname = parts[1]
                # Kick the defender by team name.
                if active_sessions["defender"] and active_sessions["defender"]["teamname"] == teamname:
                    await active_sessions["defender"]["websocket"].close()
                    active_sessions["defender"] = None
                    logging.info(f"Kicked defender: {teamname}")
                # Kick an attacker by team name.
                elif any(attacker["teamname"] == teamname for attacker in active_sessions["attackers"]):
                    attacker_to_kick = next(attacker for attacker in active_sessions["attackers"] if attacker["teamname"] == teamname)
                    await attacker_to_kick["websocket"].close()
                    active_sessions["attackers"].remove(attacker_to_kick)
                    logging.info(f"Kicked attacker: {teamname}")
                else:
                    logging.info(f"Team {teamname} not found.")
            else:
                logging.info("Invalid command syntax. Use 'kick <teamname>'.")


# Utility function to safely send messages through WebSocket
async def safe_websocket_send(websocket, message):
    try:
        await websocket.send(message)
    except Exception as e:
        logging.error(f"WebSocket send error: {str(e)}")


async def main():
    parser = argparse.ArgumentParser(description="WebSocket server for competition.")
    parser.add_argument('--api-key-file', type=validate_file, required=True, help="Path to API key file.")
        # Set default value for dataset templates file path
    parser.add_argument('--dataset-templates', type=validate_file, required=False, default='./ds-templates.jsonl', help="Path to dataset templates file.")
    parser.add_argument('--port', type=int, default=6789, help="WebSocket server port.")
    parser.add_argument('--log-level', type=str, default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'], help="Set the logging level")
    args = parser.parse_args()
    
    logging.basicConfig(level=getattr(logging, args.log_level.upper()), format='%(asctime)s - %(levelname)s - %(message)s')

    # Load user data and dataset templates, using default path if not provided
    load_user_data(args.api_key_file)
    load_dataset_templates(args.dataset_templates)


    server = websockets.serve(handler, "localhost", args.port)
    
    console_task = asyncio.create_task(console_input_handler())
    
    await asyncio.gather(server, console_task)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except asyncio.CancelledError:
        logging.info("Server shutdown.")
