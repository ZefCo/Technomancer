import pathlib
from os.path import basename
from __log_fn import setup_logs
import logging
logger = logging.getLogger(__name__)
setup_logs(pathlib.Path(basename(__file__)).stem)

import gradio as gr

import ollama

import re

import subprocess

import yaml

cwd = pathlib.Path.cwd()




# Add logging here to make sure things are being added properly.
def append_state_list(states, state):
    '''
    Updates a state list with a new stat added.
    '''
    if state and state not in tuple(states):
        states.append(state)
    elif not state and state not in tuple(states):
        states.append("Generic")
    
    states.sort()
    return states


def chatbot_avatars(user, bot):
    '''
    Gives the user and the chatbot an avatar
    '''
    if user: user = str(cwd / "Images" / user)
    if bot: bot = str(cwd / "Images" / bot)

    return gr.Chatbot(avatar_images = [user, bot])


# Adding logging here to make sure old state is being changed to new state.
def change_state_list(state_list):
    '''
    Changes a state list to another state.
    '''
    return state_list


def change_state(new_state, old_state = None, 
                 log_info = False, state_name = "State"):
    '''
    Changes the state.
    '''
    if log_info: logger.info(f"{state_name} changed | Old State: {old_state} | New State: {new_state}")
    return new_state


def change_state_per(old_state):
    '''
    Changes a state to be 0.1 of the old state. Useful for the chunk overlap
    '''
    return int(0.1 * old_state)


def check_path(paths: list):
    '''
    Checks to make sure the path is a valid path and exists. Transforms it to a string and returns the list.
    '''
    # add logging to this. It writes to the logs which paths were input and which ones were written
    path: pathlib.Path
    return_paths = list()
    for path in paths:
        try:
            path.mkdir(parents = True, exist_ok = False)
        except FileExistsError as e:
            logger.info(f"Found already existsing directory | {path}")
        else:
            logger.info(f"Created new path to folder | {path}")

        return_paths.append(str(path))

    return return_paths


def _extract_content(content) -> str:
    '''
    Gradio 6 changed message content to potentially be a list of content blocks.
    This normalizes it back to a plain string regardless of which format arrives.
    '''
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        # extract text from each block and join
        return " ".join(
            block.get("text", "") if isinstance(block, dict) else str(block)
            for block in content
        )
    return str(content)


def find_models():
    '''
    Finds all the models that are installed on the computer. Meant to be run when the server starts.
    '''
    result = subprocess.run(['ollama', 'list'], capture_output = True, text = True)
    if len(result.stderr) > 0: logger.critical(f"Errors when pulling Ollama modes | {result.stderr}")
    # Shut down system? Raise error?
    
    lines = result.stdout.strip().splitlines()[1:]  # Skip header line
    
    models = [line.split()[0] for line in lines]  # Get model names from the first column
    logger.info(f"Found Models at runtime | {models}")
    
    return models


# Logging should be here, but not sure how. Esepcially since this is related to the query routing.
def generate_response(message, history, lang_model, embed_model, 
                      collection = None, use_rag = False,
                      *args, **kwargs):
    '''
    This will route between RAG and the Ollama chat depending on the if a collection is used or not, which hopefully is triggered properly in the context of the message.
    '''
    if use_rag and collection:
        from __rag_pipeline import query_rag
        yield from query_rag(message, history, collection, lang_model, embed_model) # type: ignore
    else:
        messages = []

        for item in history:
            content = _extract_content(item["content"])
            if content: messages.append({"role": item["role"], "content": content})  # appends the information into the message with the proper format

        messages.append({"role": "user", "content": message})

        completion = ollama.chat(model = lang_model, messages = messages, stream = True)

        response = ""
        for chunk in completion:
            if "message" in chunk and "content" in chunk["message"]:
                response += chunk["message"]["content"]
                yield response


def import_setting(setting_file):
    '''
    Import a settings file
    '''
    logger.info(f"Loading settings | File {setting_file}")
    with open(cwd / "Settings" / setting_file, "r") as file:
        settings = yaml.safe_load(file)
    logger.info(f"Loaded settings | {settings}")

    return settings


def load_paths():
    '''
    Loads all the Database paths the user has saved in the settings file.
    '''
    # Add logging here to find which places are added.
    paths = import_setting("DB_Paths.yaml")

    DBoH = cwd / paths["default_path"]
    alternatives = paths["alternative_paths"]
    if alternatives: all_paths = [DBoH] + alternatives
    else: all_paths = [DBoH]

    all_paths = check_path(all_paths)

    return all_paths


def load_tags():
    '''
    Loads the metadata tags
    '''
    tags = import_setting("Tags.yaml")
    return tags["tags"]


def sort_models():
    '''
    Sorts them between embedding and language models. Assumes - and this might be a dangerous assumption - that embedding models contain the word embedding.
    '''
    # Add logging here to see which models were sorted to where.
    language: list = []
    embedding: list = []
    models = find_models()

    for model in models:
        if re.search(r"embedding", model): embedding.append(model)
        else: language.append(model)

    logger.info(f"Language models found | {language}")
    logger.info(f"Embedding models found | {embedding}")

    return language, embedding


def stop_command():
    '''
    Halts the chat.
    '''
    return False


# Add logging here, reporting the model and embedding used.
def technomancer_response(chat_history, lang_model, embed_model, *args, **kwargs):
    '''
    This yields the chatbots response. Again, *args and **kwargs are not really needed, but are here *in case*
    '''
    raw_user_msg = chat_history[-2]["content"]  # grab the content of the users message
    user_msg = _extract_content(raw_user_msg)
    chat_history[-2]["content"] = user_msg

    for response in generate_response(user_msg, chat_history[:-2], lang_model, embed_model, *args, **kwargs):
        chat_history[-1]["content"] = response  # now the last item is the assistant placeholder
        yield chat_history



# def update_chunk(old_chunk, new_chunk):
#     '''
#     Will update both chunks and overlap as the slider is adjusted.
#     '''
#     logger.info(f"Updating")
#     return chunk


def update_drop_down(choices):
    '''
    Updates a drop down menu. Useful for rule systems, documents, and model choices
    '''
    if choices: return gr.Dropdown(choices = choices, value = choices[0] if choices[0] else None)
    else: return gr.Dropdown(choices = [])


def update_slider(value, percent = 1):
    '''
    Updates the slider value
    '''
    return gr.Slider(value = int(percent * value))


def update_system_prompt(new_prompt, current):
    '''
    Only updates if the new prompt is not empty.
    Returns the updated state and refreshes the display.
    '''
    if new_prompt.strip():
        return new_prompt.strip(), new_prompt.strip(), ""
    return current, current, ""


def update_textbox(message):
    '''
    Updates the textbox.
    '''
    return gr.Textbox(placeholder = message)


def user_submit(user_msg, chat_history, *args, **kwargs):
    user_msg = _extract_content(user_msg)  # normalize incoming content
    chat_history.append({"role": "user", "content": user_msg})
    chat_history.append({"role": "assistant", "content": ""})
    return "", chat_history