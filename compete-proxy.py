import argparse
import asyncio
import logging
import pandas as pd
import websockets
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Global variables
user_data = {}
active_sessions = {"defender": None, "attackers": []}

def validate_file(file_path):
    if not Path(file_path).is_file():
        raise argparse.ArgumentTypeError(f"The file {file_path} does not exist.")
    return file_path

def load_user_data(api_key_file):
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
            return user_data[teamname]['role'], teamname
        else:
            await websocket.send("Authentication failed.")
            return None, None
    except Exception as e:
        logging.error(f"Authentication error: {e}")
        return None, None

async def handler(websocket, path):
    role, teamname = await authenticate(websocket)
    if role == "defender":
        if active_sessions["defender"] is None:
            active_sessions["defender"] = websocket
            logging.info(f"Defender {teamname} connected.")
            await websocket.send("You are connected as the defender.")
            await defender_handler(websocket, teamname)
        else:
            await websocket.send("Another defender is already connected.")
    elif role == "attacker":
        active_sessions["attackers"].append(websocket)
        logging.info(f"Attacker {teamname} connected.")
        await websocket.send("You are connected as an attacker.")
        await attacker_handler(websocket, teamname)
    else:
        await websocket.send("Invalid role or authentication failed.")

async def defender_handler(websocket, teamname):
    try:
        async for message in websocket:
            # Process messages from the defender
            logging.info(f"Message from defender {teamname}: {message}")
            # Include logic specific to defenders here
    finally:
        active_sessions["defender"] = None
        logging.info(f"Defender {teamname} disconnected.")

async def attacker_handler(websocket, teamname):
    try:
        async for message in websocket:
            # Process messages from attackers
            logging.info(f"Message from attacker {teamname}: {message}")
            # Include logic specific to attackers here
    finally:
        active_sessions["attackers"].remove(websocket)
        logging.info(f"Attacker {teamname} disconnected.")

def main():
    parser = argparse.ArgumentParser(description="Starts the contest proxy server.")
    parser.add_argument('--api-key-file', type=validate_file, required=True, help="Path to the API key file.")
    parser.add_argument('--config-file', type=validate_file, required=True, help="Path to the benchmark configuration file.")
    parser.add_argument('--port', type=int, default=6789, help="Server port.")

    args = parser.parse_args()
    load_user_data(args.api_key_file)

    start_server = websockets.serve(handler, "localhost", args.port)
    logging.info(f"Server starting on port {args.port}...")
    asyncio.get_event_loop().run_until_complete(start_server)
    asyncio.get_event_loop().run_forever()

