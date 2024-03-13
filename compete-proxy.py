import argparse
import asyncio
import logging
import pandas as pd
import websockets
from pathlib import Path

# Setup logging for detailed diagnostics
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Global variables for user data and session management
user_data = {}
active_sessions = {"defender": None, "attackers": []}

# Simulated prompt queue and score tracking for demonstration
prompt_queue = asyncio.Queue()
scores = {}
[prompt_queue.put_nowait(f"Prompt #{i}") for i in range(1, 101)]  # Load example prompts

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
    """Authenticate users and initialize their session."""
    try:
        auth_message = await websocket.recv()
        teamname, apikey = auth_message.split()
        if teamname in user_data and user_data[teamname]['apikey'] == apikey:
            role = user_data[teamname]['role']
            session = {"role": role, "teamname": teamname, "state": SessionState.Unauthenticated, "websocket": websocket}
            return session
        else:
            await websocket.send("Authentication failed.")
            return None
    except Exception as e:
        logging.error(f"Authentication error: {e}")
        await websocket.send("Authentication error. Please check the format of your authentication message.")
        return None

async def process_command(websocket, session, command):
    """Process commands based on session state and user role."""
    parts = command.strip().split()
    command_type = parts[0].lower()

    if command_type == "compete":
        response = await toggle_compete_mode(session)
    elif command_type == "^c":
        response = await exit_compete_mode(session)
    elif command_type == "login":
        response = "Already logged in."
    elif command_type == "logout":
        response = await handle_logout(session)
    elif command_type == "score":
        response = await display_scores(session)
    elif command_type in ["receive-prompt", "p"]:
        response = await handle_receive_prompt(session)
    elif command_type in ["submit-answer", "s"]:
        response = await handle_submit_answer(session, parts[1:])
    elif command_type in ["request-token", "r"]:
        response = "Token requested. Placeholder response."
    elif command_type in ["submit-attack", "a"]:
        response = await handle_submit_attack(session, parts[1:])
    else:
        response = "Unknown command or malformed input."

    await websocket.send(response)

async def toggle_compete_mode(session):
    """Toggle between LoggedIn and Compete states."""
    # Toggle logic to switch states
    if session['state'] in [SessionState.LoggedInWaiting, SessionState.LoggedInResponding]:
        session['state'] = SessionState.CompeteWaiting if session['state'] == SessionState.LoggedInWaiting else SessionState.CompeteResponding
        return "Entered compete REPL mode."
    elif session['state'] in [SessionState.CompeteWaiting, SessionState.CompeteResponding]:
        session['state'] = SessionState.LoggedInWaiting if session['state'] == SessionState.CompeteWaiting else SessionState.LoggedInResponding
        return "Exited compete REPL mode."
    return "Compete mode toggle failed due to invalid session state."

async def exit_compete_mode(session):
    """Exit Compete mode and revert to LoggedIn state."""
    if session['state'] in [SessionState.CompeteWaiting, SessionState.CompeteResponding]:
        session['state'] = SessionState.LoggedInWaiting
        return "Exited compete mode."
    return "Not in compete mode."

async def handle_logout(session):
    """Handle user logout."""
    # Reset the session state to Unauthenticated
    session['state'] = SessionState.Unauthenticated
    return "Logged out successfully."

async def display_scores(session):
    """Display the user's or team's score."""
    # Placeholder for displaying scores
    score = scores.get(session['teamname'], 0)
    return f"Your score: {score}"

async def handle_receive_prompt(session):
    """Send the next prompt to the defender."""
    if session['role'] != "defender":
        return "ERROR: Only defender sessions can receive prompts."
    if session['state'] not in [SessionState.CompeteWaiting, SessionState.LoggedInWaiting]:
        return "ERROR: Not in the correct state to receive a prompt."
    prompt = await prompt_queue.get()
    session['state'] = SessionState.CompeteResponding if session['state'] == SessionState.CompeteWaiting else SessionState.LoggedInResponding
    return prompt

async def handle_submit_answer(session, answer_parts):
    """Process the submission of an answer by a defender."""
    if session['role'] != "defender":
        return "ERROR: Only defender sessions can submit answers."
    if session['state'] not in [SessionState.CompeteResponding, SessionState.LoggedInResponding]:
        return "ERROR: Not in the correct state to submit an answer."
    answer = " ".join(answer_parts)
    # Placeholder logic for answer submission and scoring
    score = len(answer) % 10  # Example scoring mechanism
    scores[session['teamname']] = scores.get(session['teamname'], 0) + score
    session['state'] = SessionState.CompeteWaiting if session['state'] == SessionState.CompeteResponding else SessionState.LoggedInWaiting
    return f"Answer received. Score: {score}"

async def handle_submit_attack(session, parts):
    """Process the submission of an attack by an attacker."""
    if session['role'] != "attacker":
        return "ERROR: Only attacker sessions can submit attacks."
    if session['state'] not in [SessionState.CompeteResponding, SessionState.LoggedInResponding]:
        return "ERROR: Not in the correct state to submit an attack."

    # Parsing attack command options
    prefix, postfix, attach_to = None, None, None
    for i, part in enumerate(parts):
        if part == "--prefix" and i + 1 < len(parts):
            prefix = parts[i + 1]
        elif part == "--postfix" and i + 1 < len(parts):
            postfix = parts[i + 1]
        elif part == "--attach-to" and i + 1 < len(parts):
            attach_to = parts[i + 1]

    if not attach_to or (not prefix and not postfix):
        return "ERROR: Invalid attack command. Must specify --attach-to with either --prefix or --postfix."

    # Placeholder logic for attack submission
    session['state'] = SessionState.CompeteWaiting if session['state'] == SessionState.CompeteResponding else SessionState.LoggedInWaiting
    return f"Attack submitted with prefix: {prefix}, postfix: {postfix}, attached to: {attach_to}."

async def handler(websocket, path):
    """Main handler for incoming websocket connections."""
    session = await authenticate(websocket)
    if session:
        if session["role"] == "defender" and not active_sessions["defender"]:
            active_sessions["defender"] = websocket
            session['state'] = SessionState.LoggedInWaiting  # Set initial state for authenticated defender
        elif session["role"] == "attacker":
            active_sessions["attackers"].append(websocket)
            session['state'] = SessionState.LoggedInWaiting  # Set initial state for authenticated attackers
        else:
            await websocket.send("Another defender is already connected.")
            return

        try:
            async for message in websocket:
                await process_command(websocket, session, message)
        finally:
            if session["role"] == "defender":
                active_sessions["defender"] = None
            else:
                active_sessions["attackers"].remove(websocket)
    else:
        await websocket.send("Invalid role or authentication failed.")

def main():
    """Main function to start the WebSocket server."""
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
