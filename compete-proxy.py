
import argparse
import asyncio
import logging
import pandas as pd
import websockets
import aioconsole
from pathlib import Path

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
    parser.add_argument('--port', type=int, default=6789, help="WebSocket server port.")
    args = parser.parse_args()
    
    load_user_data(args.api_key_file)
    server = websockets.serve(handler, "localhost", args.port)
    
    console_task = asyncio.create_task(console_input_handler())
    
    await asyncio.gather(server, console_task)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except asyncio.CancelledError:
        logging.info("Server shutdown.")
