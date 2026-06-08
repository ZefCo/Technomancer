from dotenv import load_dotenv
import inspect
import logging
import pathlib

cwd = pathlib.Path.cwd()
logger = logging.getLogger(__name__)

import gradio as gr

# import ollama
import os

import re

import subprocess

import time

import yaml

# logger = logging.getLogger(__name__)




# Add logging here to make sure things are being added properly.
def append_state_list(states: list, state: list | str):
    '''
    Updates a state list with a new stat added.
    '''
    logger.info(f"{state} | {states} | {type(state)}")
    if isinstance(state, str): states = set([state]) | set(states)
    elif isinstance(state, list): states = set(state) | set(states)
    states = list(states)

    # if isinstance(state, str):
    #     if state and state not in tuple(states):
    #         states.append(state)
    #     elif not state and state not in tuple(states):
    #         states.append("Generic")

    # else:
    #     for s in state:
    #         if s not in tuple(states):
    #             states.append(state)

    states.sort()
    logger.info(f"List is now {states}")
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


def change_state(new_state: str, old_state: str = None | str, 
                 log_info: bool = False, state_name: str = "State"):
    '''
    Changes the state.
    '''
    if log_info:
        caller_frame = inspect.currentframe().f_back
        caller_code = caller_frame.f_code

        func_name = caller_code.co_name
        file_name = caller_code.co_filename
        logger.info(f"{file_name} | {func_name} | {state_name} changed | Old State: {old_state} | New State: {new_state}")
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


def _enter_event(value, key_data: gr.KeyUpData):
    '''
    Updates something when the enter key is pressed.
    '''
    if key_data.key == "Enter":
        return value
    

def export_tags(tags: list):
    '''
    Exports the settings to a save file
    '''
    tags = {"tags": tags}
    settings_path = cwd / "Settings" / "Tags.yaml"
    try:
        with open(settings_path, "w") as file:
            yaml.dump(tags, file)
    except Exception as e:
        logger.info(f"Error exporting Metadata Tags | {settings_path} | {tags} | {type(e)} | {e}")
    else:
        logger.info(f"Successfully save Metadata Tags")


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
    result = None
    try:
        result = subprocess.run(['ollama', 'list'], capture_output = True, text = True)
    except Exception as e:
        logger.critical(f"Subprocess command to Ollama failed | {type(result)}")
        raise FileNotFoundError
    if len(result.stderr) > 0: logger.critical(f"Errors when pulling Ollama modes | {result.stderr}")
    # Shut down system? Raise error?
    
    lines = result.stdout.strip().splitlines()[1:]  # Skip header line
    
    models = [line.split()[0] for line in lines]  # Get model names from the first column
    logger.info(f"Found Models at runtime | {models}")
    
    return models


# # Logging should be here, but not sure how. Esepcially since this is related to the query routing.
# def generate_response(message, history, lang_model, embed_model, 
#                       collection: str | None = None, use_rag = True,
#                       *args, **kwargs):
#     '''
#     This will route between RAG and the Ollama chat depending on the if a collection is used or not, which hopefully is triggered properly 
#     in the context of the message.
    
#     Because of scope creep, I'm hard coding this to ALWAYS use RAG. Honeslty, while it would be useful to maybe use query routing one day, 
#     Technomancer doesn't need it. I'm leaving the option of one day comeing back to it, or if someone clones this and decided they want to
#     implement this themselves, they can. But a) I will not try to fix any errors they come across (sorry, if you're reading this, you're on
#     you're own right now), and b) this project doesn't really require or demand it.
#     '''
#     if use_rag and collection:
#         from __rag_pipeline import query_rag_routed
#         yield from query_rag_routed(message, history, collection, lang_model, embed_model) # type: ignore
#     else:
#         messages = []

#         for item in history:
#             content = _extract_content(item["content"])
#             if content: messages.append({"role": item["role"], "content": content})  # appends the information into the message with the proper format

#         messages.append({"role": "user", "content": message})

#         completion = ollama.chat(model = lang_model, messages = messages, stream = True)

#         response = ""
#         for chunk in completion:
#             if "message" in chunk and "content" in chunk["message"]:
#                 response += chunk["message"]["content"]
#                 yield response


def import_evn_settings():
    '''
    Imports all the settings from the .env settings file.

    This is in the testing phase. It's fairly straight forward, but the advantage of .yaml files is that they are easy to adjust and are human readable.
    They can also be broken up. Should the .env files be broken up? Should they be kept together? Are they easy to shove into the states? Can they easily
    be written to? Answer is probably yes to most of them.
    '''
    load_dotenv(str(cwd.parent / "Settings" / "SETTINGS.env"))

    MY_ENV_TEST = os.getenv("AVATARS", None)


def import_setting(setting_file):
    '''
    Import a settings file
    '''
    caller_frame = inspect.currentframe().f_back
    caller_code = caller_frame.f_code

    func_name = caller_code.co_name
    file_name = caller_code.co_filename
    
    logger.info(f"{file_name} | {func_name} | Loading settings | File {setting_file}")
    with open(cwd / "Settings" / setting_file, "r") as file:
        settings = yaml.safe_load(file)
    logger.info(f"Loaded settings | {settings}")

    return settings


def live_stream(file_path: pathlib.Path | str):
    '''
    Live streams a file, mostly for the logs. That way the logs can be read in real time.
    '''
    with open(file_path, "r") as logfile:
        logfile.seek(0)
        log_text = ""

        while True:
            line = logfile.readline()
            if not line:
                time.sleep(1)
                continue

            log_text += line
            yield log_text


def list_length(lst):
    '''
    For returning the length of a list. It's really used just to count the number of documents in the collection.
    '''
    return len(lst) if lst else 0


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
    Sorts them between embedding and language models. There is a (non comprehensive) list of embedding models, and if one of them is found it is tagged as an
    embedding mode. Again, this is non comprehensive, so that list may need to be adjusted on a per user basis.

    This iterates through all available models tries to match them to an embedding model. If it does, then it appends it to the embedding model list. If it doesn't
    then it appends it to the langauge model list.
    '''
    embedding_models = import_setting("EmbeddingModels.yaml")
    embedding_models = tuple(embedding_models["models"])
    
    language: list = []
    embedding: list = []
    try:
        models = find_models()
    except FileNotFoundError as e:
        logger.critical(f"Check if Ollama is installed and running")
        raise FileNotFoundError
    except Exception as e:
        logger.critical(f"New error | {type(e)} | {e}")
        raise type(e)

    for model in models:
        for em in embedding_models:
            if re.search(em, model): 
                embedding.append(model)
                break
        else: language.append(model)

    logger.info(f"{language} | {embedding}")

    # if len(language) < 1: 
    if language is None:
        logger.critical(f"No Language models identified | {models}")
        print("No Language models found, check logs and system for models")
    else: 
        logger.info(f"Language models found | {language}")
    
    # if len(embedding) < 1:
    if embedding is None: 
        logger.critical(f"No Embedding models identified | {models}")
        print("No Embedding models found, check logs and system for models")
    else: 
        logger.info(f"Embedding models found | {embedding}")

    return language, embedding


def stop_command():
    '''
    Halts the chat.
    '''
    return False


def technomancer_response(chat_history, lang_model, embed_model,
                          hr_collection: str | None = None, tags: list[str] | None = None, k: int = 10, score_threshold: float = 0.3,
                          *args, **kwargs):
    '''
    This yields the chatbots response. Again, *args and **kwargs are not really needed, nor used, because Gradio doesn't allow it.
    '''
    raw_user_msg = chat_history[-2]["content"]  # grab the content of the users message
    user_msg = _extract_content(raw_user_msg)
    chat_history[-2]["content"] = user_msg

    from __rag_pipeline import query_rag_routed
    logger.info(f"Generating response | Lang Model {lang_model} | Embed Model {embed_model}")
    
    # for response in query_rag_routed(user_msg, chat_history[:-2], lang_model, embed_model, collection=hr_collection, tags=tags, *args, **kwargs):
    for response in query_rag_routed(user_msg, chat_history[:-2], lang_model, embed_model, collection=hr_collection, tags=tags, k = k, score_threshold = score_threshold):
        chat_history[-1]["content"] = response  # now the last item is the assistant placeholder
        yield chat_history



# def update_chunk(old_chunk, new_chunk):
#     '''
#     Will update both chunks and overlap as the slider is adjusted.
#     '''
#     logger.info(f"Updating")
#     return chunk


def update_drop_down(choices: list, choice: str | None = None):
    '''
    Updates a drop down menu. Useful for rule systems, documents, and model choices
    '''
    if choices and choice: return gr.Dropdown(choices = choices, value = choice)
    elif choices: return gr.Dropdown(choices = choices)
    else: return gr.Dropdown(choices = [])


def update_number(value):
    '''
    Updates the number in the number box
    '''
    return gr.Number(value = value)


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
    return gr.Textbox(value = message)


def update_textbox_label(label):
    '''
    Changes the label of the textbox, i.e. when selecting a log file, makes the label show the log file name
    '''
    return gr.Textbox(label = label)


def user_submit(user_msg, chat_history, *args, **kwargs):
    user_msg = _extract_content(user_msg)  # normalize incoming content
    chat_history.append({"role": "user", "content": user_msg})
    chat_history.append({"role": "assistant", "content": ""})
    return "", chat_history


# if __name__ in "__main__":
#     setup_logs(pathlib.Path(basename(__file__)).stem)

print("Finished reading Functions file")
logger.info(f"Finished reading Gradio functions file")

