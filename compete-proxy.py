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
            await safe_websocket_send(websocket, json.dumps({"id": prompt_id, "prompt": prompt_text}))
        else:
            await safe_websocket_send(websocket, "ERROR: No more prompts available.")
    except StopIteration:
        await safe_websocket_send(websocket, "ERROR: No more prompts available.")
    except Exception as e:
        logging.error(f"Error handling receive prompt request: {str(e)}")
        await safe_websocket_send(websocket, "ERROR: An internal error occurred while processing your request.")
