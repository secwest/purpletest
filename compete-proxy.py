import argparse
import asyncio
import logging
import pandas as pd
import websockets
from pathlib import Path

# Set up detailed logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Global variables for user data and session management
user_data = {}
active_sessions = {"defender": None, "attackers": []}

# Simulated prompt queue for demonstration
prompt_queue = asyncio.Queue()
# Simulate loading prompts into the queue
[prompt_queue.put_nowait(f"Prompt #{i}") for i in range(1, 101)]

class SessionState:
    LoggedInWaiting = "LoggedInWaiting"
    LoggedInResponding = "LoggedInResponding"
    CompeteWaiting = "CompeteWaiting"
    CompeteResponding = "CompeteResponding"
    Unauthenticated = "Unauthenticated"

def validate_file(file_path):
    """Validate if the specified file exists."""
    if not Path(file_path).is_file():
        raise argparse.ArgumentTypeError(f"The file {file_path} does not exist.")
    return file_path

def load_user_data(api_key_file):
    """Load user data from a CSV file into a dictionary."""
    try:
        data = pd.read_csv(api_key_file)
        for _, row in data.iterrows():
            user_data[row['teamname']] = {'apikey': row['apikey'], 'role': row['role']}
        logging.info("User data loaded successfully.")
    except Exception as e:
        logging.error(f"Failed to load user data: {e}")
        exit(1)

async def authenticate(websocket):
    try:
        auth_message = await websocket.recv()
        teamname, apikey = auth_message.split()
        if teamname in user_data and user_data[teamname]['apikey'] == apikey:
            role = user_data[teamname]['role']
            state = SessionState.LoggedInWaiting if role == "defender" else SessionState.Unauthenticated
            session = {"role": role, "teamname": teamname, "state": state}
            return session
        else:
            await websocket.send("Authentication failed.")
            return None
    except Exception as e:
        logging.error(f"Authentication error: {e}")
        await websocket.send("Authentication error. Please check the format of your authentication message.")
        return None

async def process_attack_command(session, parts):
    if session['role'] != "attacker":
        return "Error: Only attacker sessions can submit attacks."

    if session['state'] not in [SessionState.CompeteResponding, SessionState.LoggedInResponding]:
        return "Illegal Command For Connection State"

    options = {"prefix": None, "postfix": None, "attach-to": None}
    main_query = None

    iter_parts = iter(parts[1:])  # Skip the command itself
    for part in iter_parts:
        if part.startswith("--"):
            option = part[2:]
            if option in options:
                options[option] = next(iter_parts, None)
            else:
                logging.warning(f"Unknown option {part} ignored.")
        else:
            main_query = part

    if not main_query and all(options.values()):
        session['state'] = SessionState.CompeteWaiting if session['state'] == SessionState.CompeteResponding else SessionState.LoggedInWaiting
        logging.info(f"Attack submitted by {session['teamname']} with options: {options}")
        return f"Attack submitted successfully with options: {options}."
    else:
        return "Error: Invalid command syntax for submit-attack."

async def process_command(websocket, session, command):
    parts = command.strip().split()
    command_type = parts[0].upper()

    if not parts or command_type not in ["RECEIVE-PROMPT", "P", "SUBMIT-ANSWER", "S", "SUBMIT-ATTACK", "A"]:
        logging.debug(f"Received unknown or malformed command: '{command}'")
        await websocket.send("ERROR: Unknown command or malformed input.")
        return

    response = ""
    try:
        if command_type in ["RECEIVE-PROMPT", "P"]:
            if session['role'] == "defender":
                prompt = await prompt_queue.get()
                session['state'] = SessionState.LoggedInResponding
                response = prompt
            else:
                response = "Error: Only defender sessions can receive prompts."
        elif command_type in ["SUBMIT-ANSWER", "S"]:
            if session['role'] == "defender":
                answer = " ".join(parts[1:])
                score = f"Score for your answer '{answer}': {len(answer) % 10}/10"  # Placeholder score calculation
                session['state'] = SessionState.LoggedInWaiting
                response = score
            else:
                response = "Error: Only defender sessions can submit answers."
        elif command_type in ["SUBMIT-ATTACK", "A"]:
            response = await process_attack_command(session, parts)
        else:
            response = "ERROR: Unknown command or not allowed in current state."
        
        if response:
            await websocket.send(response)
    except Exception as e:
        logging.error(f"Error processing command '{command}': {e}")
        await websocket.send("An error occurred processing your command. Please try again.")

async def handler(websocket, path):
    session = await authenticate(websocket)
    if session:
        if session["role"] == "defender":
            active_sessions["defender"] = websocket
        else:
            active_sessions["attackers"].append(websocket)
        try:
            async for message in websocket:
                await process_command(websocket, session, message)
        finally:
            if session["role"] == "defender":
                active_sessions["defender"] = None
            else:
                active_sessions["attackers"].remove(websocket)
            logging.info(f"{session['role'].capitalize()} {session['teamname']} disconnected.")
    else:
        await websocket.send("Invalid role or authentication failed.")

def main():
    parser = argparse.ArgumentParser(description="Starts the contest proxy server.")
    parser.add_argument('--api-key-file', type=validate_file, required=True, help="Path to the API key file.")
    parser.add_argument('--port', type=int, default=6789, help="Server port.")
    args = parser.parse_args()
    load_user_data(args.api_key_file)
    start_server = websockets.serve(handler, "localhost", args.port)
    asyncio.get_event_loop().run_until_complete(start_server)
    asyncio.get_event_loop().run_forever()

if __name__ == "__main__":
    main()
