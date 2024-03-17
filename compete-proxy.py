
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
                template = json.loads(line.strip())
                key = f"{template['dataset']}_{template['config']}"

                # Assuming 'answer_fields' can now be a list of fields
                # and 'token_join_fields' specifies how to join tokens for evaluation
                dataset_templates[key] = {
                    "template": template["template"],
                    "answer_fields": template.get("answer_fields", []),  # List of fields
                    "answer_type": template["answer_type"],
                    "evaluator_type": template["evaluator_type"],
                    "token_join_fields": template.get("token_join_fields", {}),  # How to join tokens
                    "scoring_data": template.get("scoring_data", {})  # Includes weights or other scoring parameters
                }
    except FileNotFoundError:
        logging.error(f"The file {file_path} was not found.")
    except json.JSONDecodeError as e:
        logging.error(f"Error parsing JSON: {e}")


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

async def evaluate_answer(evaluator_type, submitted_answer, correct_answers, scoring_data):
    """Dispatches the submitted answer to the appropriate evaluator."""
    evaluators = {
        "exact_match": exact_match_evaluator,
        "numeric": numeric_evaluator,
        "f1_score": f1_score_evaluator,
        "bleu_score": bleu_score_evaluator,
        "accuracy": accuracy_evaluator,
        "precision": precision_evaluator,
        "recall": recall_evaluator,
        "pearson_correlation": pearson_correlation_evaluator,
        "spearman_correlation": spearman_correlation_evaluator
    }

    evaluator = evaluators.get(evaluator_type)
    if evaluator:
        # Handle evaluators expecting different parameters gracefully
        if evaluator_type in ["exact_match", "f1_score", "bleu_score"]:
            return evaluator(submitted_answer, correct_answers)
        elif evaluator_type in ["numeric", "accuracy", "precision", "recall", "pearson_correlation", "spearman_correlation"]:
            return evaluator(submitted_answer, correct_answers, scoring_data)
    else:
        logging.error(f"Evaluator type '{evaluator_type}' not supported.")
        return 0




def exact_match_evaluator(submitted_answer, correct_answers):
    """Checks if submitted answer exactly matches any of the correct answers."""
    return int(submitted_answer in correct_answers)

def numeric_evaluator(submitted_answer, correct_answers, scoring_data):
    """Evaluates numeric answers with optional tolerance."""
    try:
        submitted_value = float(submitted_answer)
        correct_value = float(correct_answers[0])
        tolerance = scoring_data.get('tolerance', 0)
        return int(abs(submitted_value - correct_value) <= tolerance)
    except (ValueError, TypeError):
        return 0

def f1_score_evaluator(submitted_answer, correct_answers):
    """Computes the F1 score for the submitted answer against the correct answers."""
    # Tokenization or splitting should be handled as needed for your specific use case
    submitted_tokens = submitted_answer.split()
    correct_tokens = [ans.split() for ans in correct_answers]
    scores = [f1_score([correct], [submitted_tokens], average='macro') for correct in correct_tokens]
    return max(scores)

def bleu_score_evaluator(submitted_answer, correct_answers):
    """Calculates the BLEU score for the submitted answer against the correct answers."""
    # Tokenization or splitting should be handled as needed for your specific use case
    submitted_tokens = submitted_answer.split()
    correct_tokens = [ans.split() for ans in correct_answers]
    score = sentence_bleu(correct_tokens, submitted_tokens)
    return score

def accuracy_evaluator(submitted_answer, correct_answers):
    """Calculates accuracy assuming the submitted answer is among the correct answers."""
    return accuracy_score(correct_answers, [submitted_answer])

def precision_evaluator(submitted_answer, correct_answers):
    """Calculates precision for classification tasks."""
    return precision_score(correct_answers, [submitted_answer], average='macro')

def recall_evaluator(submitted_answer, correct_answers):
    """Calculates recall for classification tasks."""
    return recall_score(correct_answers, [submitted_answer], average='macro')

def pearson_correlation_evaluator(submitted_answer, correct_answers):
    """Evaluates Pearson correlation between submitted and correct answers."""
    submitted_value = float(submitted_answer)
    correct_values = [float(ans) for ans in correct_answers]
    correlation, _ = pearsonr(correct_values, [submitted_value])
    return correlation

def spearman_correlation_evaluator(submitted_answer, correct_answers):
    """Evaluates Spearman correlation between submitted and correct answers."""
    submitted_value = float(submitted_answer)
    correct_values = [float(ans) for ans in correct_answers]
    correlation, _ = spearmanr(correct_values, [submitted_value])
    return correlation


async def evaluate_answer(evaluator_type, submitted_answer, correct_answers, scoring_data):
    # Placeholder for evaluator dispatcher logic
    if evaluator_type == "exact_match":
        return exact_match_evaluator(submitted_answer, correct_answers)
    elif evaluator_type == "numeric":
        return numeric_evaluator(submitted_answer, correct_answers, scoring_data)
    # Add other evaluators as necessary
    return 0

def exact_match_evaluator(submitted_answer, correct_answers):
    # Example of a simple exact match evaluator
    return 1 if submitted_answer in correct_answers else 0

def numeric_evaluator(submitted_answer, correct_answers, scoring_data):
    # Placeholder for a numeric comparison evaluator
    # This could involve comparing the submitted_answer to a numeric range or value
    try:
        submitted_value = float(submitted_answer)
        correct_value = float(correct_answers[0])  # Assuming a single correct numeric answer
        tolerance = scoring_data.get('tolerance', 0)  # Allow for some tolerance in the numeric comparison
        return 1 if abs(submitted_value - correct_value) <= tolerance else 0
    except (ValueError, TypeError):
        return 0
    
async def handle_submit_answer(websocket, index, submitted_answer):
    # Ensure the function is called with the correct parameters:
    # `websocket` - the WebSocket connection of the defender
    # `index` - the index of the prompt being answered
    # `submitted_answer` - the answer provided by the defender

    # Check if the prompt exists and has not yet been answered
   if index not in unanswered_prompts:
        return "ERROR: Prompt index not found or already answered."

    prompt_info = unanswered_prompts[index]
    evaluator_type = prompt_info['template']['evaluator_type']
    scoring_data = prompt_info['template'].get('scoring_data', {})  # Scoring data might not always be present
    answer_fields = prompt_info['template'].get('answer_fields', [])  # Fields to extract the correct answer(s) from

    # Extract correct answers based on the specified fields in 'answer_fields'
    correct_answers = [prompt_info['data'][field] for field in answer_fields if field in prompt_info['data']]

    # Handle case where correct answers could not be extracted
    if not correct_answers:
        logging.error(f"No correct answers found for prompt index {index}. Check 'answer_fields' in template.")
        return "ERROR: No correct answers available for evaluation."

    # Call the specific evaluator based on `evaluator_type`
    score = await evaluate_answer(evaluator_type, submitted_answer, correct_answers, scoring_data)

    defender_feedback = f"Index: {index}, Answer received: '{submitted_answer}'. Score: {score}."
    await websocket.send(defender_feedback)

    # Feedback to attacker if present
    if 'attacks' in prompt_info:
        for attack in prompt_info['attacks']:
            attacker_websocket = next((attacker['websocket'] for attacker in active_sessions["attackers"] if attacker['teamname'] == attack['teamname']), None)
            if attacker_websocket:
                await attacker_websocket.send(defender_feedback + " " + json.dumps(attack))

    logging.info(defender_feedback + " " + json.dumps(prompt_info))
    
    # Remove the prompt from the list of unanswered prompts
    del unanswered_prompts[index]

    return defender_feedback


    

def prompt_generator(file_path, chunk_size=1024):
    """A generator that yields ID, prompt, and template from a file, reading in chunks."""
    buffer = ''
    with open(file_path, 'r') as file:
        while True:
            chunk = file.read(chunk_size)
            if not chunk:  # End of file
                break
            buffer += chunk
            lines = buffer.split('\n')
            for i, line in enumerate(lines[:-1]):  # Process all but the last line
                if line.startswith('Entry:'):
                    _, entry_json = line.split('Entry: ', 1)
                    entry = json.loads(entry_json.strip())
                    # Adjust these fields based on your dataset's structure
                    id = entry.get("idx")  # Assuming each entry has an 'idx' for ID
                    prompt = entry.get("prompt")  # The prompt text
                    template = entry.get("template")  # The template, if applicable
                    yield id, prompt, template
            buffer = lines[-1]  # Save the last line in case it's incomplete

    # Process any remaining buffer content after reading the last chunk
    if buffer.startswith('Entry:'):
        _, entry_json = buffer.split('Entry: ', 1)
        entry = json.loads(entry_json.strip())
    
        id = entry.get("idx")  # Assuming each entry has an 'idx' for ID
        prompt = entry.get("prompt")  # The prompt text
        data = entry.get("data")  # The template, if applicable
        template = entry.get("template")  # The template, if applicable
        yield id, prompt, data, template

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

async def handle_receive_prompt(websocket, session):
    if session['role'] != "defender":
        return "ERROR: Only defenders can receive prompts."
    prompt = await prompt_queue.get()
    return prompt

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
        insert_attach_queue(websocket, session, prefix, postfix, attach-to)
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


async def main():
    parser = argparse.ArgumentParser(description="WebSocket server for competition.")
    parser.add_argument('--api-key-file', type=validate_file, required=True, help="Path to API key file.")
        # Set default value for dataset templates file path
    parser.add_argument('--dataset-templates', type=validate_file, required=False, default='./ds-templates.jsonl', help="Path to dataset templates file.")
    parser.add_argument('--port', type=int, default=6789, help="WebSocket server port.")
    args = parser.parse_args()
    
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
