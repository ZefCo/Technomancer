import logging
logger = logging.getLogger(__name__)

import gradio as gr

import ollama

import pathlib

import re

import subprocess

import yaml

cwd = pathlib.Path.cwd()



def chatbot_avatars(user, bot):
    '''
    Gives the user and the chatbot an avatar
    '''
    if user: user = str(cwd / "Images" / user)
    if bot: bot = str(cwd / "Images" / bot)

    return gr.Chatbot(avatar_images = [user, bot])


def append_state_list(states, state):
    '''
    Updates a state list with a new stat added.
    '''
    if state and state not in tuple(states):
        states.append(state)
        states.sort()
    return states


def change_state_list(state_list):
    '''
    Changes a state list to another state.
    '''
    return state_list


def check_path(paths):
    '''
    Checks to make sure the path is a valid path and exists. Transforms it to a string and returns the list.
    '''
    path: pathlib.Path
    return_paths = list()
    for path in paths:
        path.mkdir(parents = True, exist_ok = True)
        return_paths.append(str(path))

    return return_paths



def find_models():
    '''
    Finds all the models that are installed on the computer. Meant to be run when the server starts.
    '''
    result = subprocess.run(['ollama', 'list'], capture_output=True, text=True)
    lines = result.stdout.strip().splitlines()[1:]  # Skip header line
    models = [line.split()[0] for line in lines]  # Get model names from the first column
    return models


def load_paths():
    '''
    Loads all the Database paths the user has saved in the settings file.
    '''
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


def import_setting(setting_file):
    '''
    Import a settings file
    '''
    with open(cwd / "Settings" / setting_file, "r") as file:
        settings = yaml.safe_load(file)

    return settings


def sort_models():
    '''
    Sorts them between embedding and language models. Assumes - and this might be a dangerous assumption - that embedding models contain the word embedding
    '''
    language: list = []
    embedding: list = []
    models = find_models()

    for model in models:
        if re.search(r"embedding", model): embedding.append(model)
        else: language.append(model)

    return language, embedding


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


def generate_response(message, history, model, embeddings, 
# def generate_response(message, history, model, embeddings, system_content = "", 
                      collection = None, use_rag = False,
                      *args, **kwargs):
    '''
    This will route between RAG and the Ollama chat depending on the if a collection is used or not, which hopefully is triggered properly in the context of the message.
    '''
    if use_rag and collection:
        from __rag_pipeline import query_rag
        yield from query_rag(message, history, collection, model, embeddings, system_content = system_content) # type: ignore
    else:
        # messages = [{"role": "system", "content": _extract_content(system_content)}]
        messages = []
        # if system_content: messages = [{"role": "system", "content": system_content}]
        # else: messages = []

        for item in history:
            content = _extract_content(item["content"])
            if content: messages.append({"role": item["role"], "content": content})  # appends the information into the message with the proper format

        messages.append({"role": "user", "content": message})

        completion = ollama.chat(model = model, messages = messages, stream = True)

        response = ""
        for chunk in completion:
            if "message" in chunk and "content" in chunk["message"]:
                response += chunk["message"]["content"]
                yield response



def stop_command():
    '''
    Halts the chat.
    '''
    return False


def technomancer_response(chat_history, model, embedding, *args, **kwargs):
# def technomancer_response(chat_history, system_content, model, embedding, *args, **kwargs):
    '''
    This yields the chatbots response. Again, *args and **kwargs are not really needed, but are here *in case*
    '''
    raw_user_msg = chat_history[-2]["content"]  # grab the content of the users message
    user_msg = _extract_content(raw_user_msg)
    chat_history[-2]["content"] = user_msg

    # for response in generate_response(user_msg, chat_history[:-2], model, embedding, system_content, *args, **kwargs):
    for response in generate_response(user_msg, chat_history[:-2], model, embedding, *args, **kwargs):
        chat_history[-1]["content"] = response  # now the last item is the assistant placeholder
        yield chat_history



def update_chunks(size, overlap):
    '''
    Will update both chunks and overlap as the slider is adjusted.
    '''
    return size, overlap


def update_drop_down(choices):
    '''
    Updates a drop down menu. Useful for rule systems, documents, and model choices
    '''
    if choices: return gr.Dropdown(choices = choices, value = choices[0] if choices[0] else None)
    else: return gr.Dropdown(choices = [])



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