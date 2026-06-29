from dotenv import load_dotenv
import inspect
import logging
import pathlib

cwd = pathlib.Path.cwd()
logger = logging.getLogger(__name__)

import gradio as gr

from __log_context import set_current_user

# import ollama
# import os

import re

import subprocess

import time
import toml

import yaml

# logger = logging.getLogger(__name__)
# Add logging here to make sure things are being added properly.
# logger = logger.LoggerAdapter(logger, user_info)

def append_state_list(states: list, state: list | str, request: gr.Request):
    '''
    Updates a state list with a new stat added.
    '''
    set_current_user(request.username)
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


def change_state(new_state: str | int | float, request: gr.Request,
                 old_state: str | int | float | None = None, log_info: bool = False, state_name: str = "State"):
    '''
    Changes the state.
    '''
    if log_info:
        set_current_user(request.username)
        # caller_frame = inspect.currentframe().f_back
        # caller_code = caller_frame.f_code

        # func_name = caller_code.co_name
        # file_name = caller_code.co_filename
        logger.info(f"{state_name} changed | Old State: {old_state} | New State: {new_state}")
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


# def chunking_default(chunking_options: list, request: gr.Request):
#     '''
#     Determines the types of chunking allowed.
    
#     Requires that chunks either be saved or summary saved or both. Something *has* to be saved. Though I guess there can be a debug option with just seeing how the file is being chunked. That's a good idea actually.
#     '''
#     set_current_user(request.username)
#     # if len(chunking_options) == 3:
#     #     return gr.CheckboxGroup(value = "Both")
#     # if ["Both", "Chunk", "Summary"]
#     if "Both" in tuple(chunking_options):
#         return gr.CheckboxGroup(value = "Both")


def chunking_type(chunking_options, request: gr.Request):
    '''
    Returns the boolean of what will be chunked.
    '''
    set_current_user(request.username)

    if "Both" in chunking_options:
        logger.info(f"Chunking = {True} | Summary = {True}")
        return True, True, gr.CheckboxGroup(info = "Will save both Chunks and Summary")
    if "Chunk" in tuple(chunking_options) and "Summary" in tuple(chunking_options):
        logger.info(f"Chunking = {True} | Summary = {True}")
        return True, True, gr.CheckboxGroup(info = "Will save both Chunks and Summary")
    if "Chunk" in tuple(chunking_options) and "Summary" not in tuple(chunking_options):
        logger.info(f"Chunking = {True} | Summary = {False}")
        return True, False, gr.CheckboxGroup(info = "Will save only chunks")
    if "Chunk" not in tuple(chunking_options) and "Summary" in tuple(chunking_options):
        logger.info(f"Chunking = {False} | Summary = {True}")
        return False, True, gr.CheckboxGroup(info = "Will save only summary")
    if "Chunk" not in tuple(chunking_options) and "Summary" not in tuple(chunking_options):
        logger.warning(f"Chunking = {False} | Summary = {False}")
        return False, False, gr.CheckboxGroup(info = "Warning: Nothing has been selected to be saved")



def collect_metadata(
        extraction_method: str, 
        quality_score: float, 
        page: int, 
        text_length: int,
        is_sparse: bool,
        word_len_suspicious: bool, 
        angled_ratio: float,
        embedding_used: str
        ):
    '''
    Collects and reorganizes the metadata. Things might get lost, like authors or publishers, but 
    I'm going to try to avoid that, and also minimize it by only loosing things that are not important.

    'id': 'Delta Green:Delta Green Agents Handbook:chunk:2:0', 'tags': ['Delta Green'], 'quality_pass': True, 'auto_tags': ['Delta Green'], 'ave_word_len': 5.75, 'Title': 'Delta Green Agents Handbook', 'text_density': 0.00519492308707995, 'doubled_ratio': 0.03125, 'chunk_type': 'text', 'has_images': True, 'has_images_only': False, 'word_count': 76, 'game_system': 'Delta Green', 'source': 'C:\\Users\\tokyo\\AppData\\Local\\Temp\\gradio\\8a0fe9aaf7c379f018c92555e559d27eb9b37be1fd3a9d7c7e251b8099551b58\\Delta Green Agents Handbook.pdf'}
    '''
    return {
        "extraction_method": extraction_method,
        'quality_score': quality_score, 
        "page": page,
        "text_length": text_length,
        "is_sparse": is_sparse,
        "word_len_suspicious": word_len_suspicious,
        "angled_ratio": angled_ratio,
        "embedding_used": embedding_used,
        }


def enable_prefix(lang_model: str):
    '''
    '''
    if lang_model.split(":")[0] in tuple(["nomic-embed-text", "nomic-embed-text-v2-moe"]):
        return gr.Checkbox(visible=True), gr.Text(visible=True)
    else:
        return gr.Checkbox(visible=False), gr.Text(visible=False)



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


def find_models(ollama_port):
    '''
    Finds all the models that are installed on the computer. Meant to be run when the server starts.
    '''
    import requests
    from requests.exceptions import ConnectionError as CE

    try:
        response = requests.get(f'http://ollama:{ollama_port}/api/tags')
    except CE as e:
        logger.critical(f"Failure to get ollama host, switching to localhost")

        try:
            response = requests.get(f"http://localhost:{ollama_port}/api/tags")
        except CE as e:
            logger.critical(f"Ollama is not running | Failed both docker and localhost connection request")
            raise 
        except Exception as e:
            logger.critical(f"Error when trying to connect to Ollama localhost | {type(e)} | {e}")
            raise 

    except Exception as e:
        logger.critical(f"Ollama client response failed | {type(e)} | {e}")
        raise 
    
    try:
        model_list = response.json().get('models', [])
    except Exception as e:
        logger.critical(f"Ollama model list failed | {type(e)} | {e} | {type(response)} | {response}")
        raise
    
    return model_list


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


def get_username(request: gr.Request):
    '''
    Gets the username.
    '''
    return request.username


def import_settings():
    '''
    '''
    with open(cwd / "Settings" / "Settings.toml", "r") as file:
        settings = toml.load(file)

    return settings

# def import_evn_settings():
#     '''
#     Imports all the settings from the .env settings file.

#     This is in the testing phase. It's fairly straight forward, but the advantage of .yaml files is that they are easy to adjust and are human readable.
#     They can also be broken up. Should the .env files be broken up? Should they be kept together? Are they easy to shove into the states? Can they easily
#     be written to? Answer is probably yes to most of them.
#     '''
#     load_dotenv(str(cwd.parent / "Settings" / "SETTINGS.env"))

#     MY_ENV_TEST = os.getenv("AVATARS", None)


# def import_setting(setting_file):
#     '''
#     Import a settings file
#     '''
#     # caller_frame = inspect.currentframe().f_back
#     # caller_code = caller_frame.f_code

#     # func_name = caller_code.co_name
#     # file_name = caller_code.co_filename
    
#     # logger.info(f"{file_name} | {func_name} | Loading settings | File {setting_file}")
#     logger.info(f"Loading Settings | File {setting_file}")
#     with open(cwd / "Settings" / setting_file, "r") as file:
#         settings = yaml.safe_load(file)
#     logger.info(f"Loaded settings | {settings}")

#     return settings


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


# def load_paths():
#     '''
#     Loads all the Database paths the user has saved in the settings file.
#     '''
#     # Add logging here to find which places are added.
#     paths = import_setting("DB_Paths.yaml")

#     DBoH = cwd / paths["default_path"]
#     alternatives = paths["alternative_paths"]
#     if alternatives: all_paths = [DBoH] + alternatives
#     else: all_paths = [DBoH]

#     all_paths = check_path(all_paths)

#     return all_paths


# def load_tags():
#     '''
#     Loads the metadata tags
#     '''
#     tags = import_setting("Tags.yaml")
#     return tags["tags"]


def sort_models(embedding_models, vision_models, ollama_port):
    '''
    Sorts them between embedding and language models. There is a (non comprehensive) list of embedding models, and if one of them is found it is tagged as an
    embedding mode. Again, this is non comprehensive, so that list may need to be adjusted on a per user basis.

    This iterates through all available models tries to match them to an embedding model. If it does, then it appends it to the embedding model list. If it doesn't
    then it appends it to the langauge model list.
    '''
    # embedding_models = import_setting("EmbeddingModels.yaml")
    # embedding_models = tuple(embedding_models["models"])
    
    language: list = []
    embedding: list = []
    vision: list = ["NONE"]  # in case someone doesn't want to use the vision model at all.
    try:
        models: list[dict] = find_models(ollama_port)
    except FileNotFoundError as e:
        logger.critical(f"Check if Ollama is installed and running")
        raise FileNotFoundError
    except Exception as e:
        logger.critical(f"New error | {type(e)} | {e}")
        raise
    
    # logger.warning(f"Models pulled | {models}")

    for model in models:
        if any(em in model["name"] for em in embedding_models): embedding.append(model["name"])
        else: language.append(model["name"])

        if any(vi in model["name"] for vi in vision_models): vision.append(model["name"])  # it can be a vision AND a language model... I think

        # for em in embedding_models:  # why not do a if model["name"] in tuple(embedding_models)? Because the names in Ollama have extra things, like the version or :latest, and that causes the search to fail.
        #     if re.search(em, model["name"]): 
        #         embedding.append(model["name"])  
        #         break


    logger.info(f"{language} | {embedding} | {vision}")

    # if language is None:
    if len(language) < 1: 
        logger.critical(f"No Language models identified | {models}")
        print("No Language models found, check logs and system for models")
    else: 
        logger.info(f"Language models found | {language}")
    
    # if embedding is None: 
    if len(embedding) < 1:
        logger.critical(f"No Embedding models identified | {models}")
        print("No Embedding models found, check logs and system for models")
    else: 
        logger.info(f"Embedding models found | {embedding}")

    if len(vision) < 1:
        logger.warning(f"No Vision models identified | {vision}")
    else:
        logger.info(f"Vision models found | {vision}")

    return language, embedding, vision


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


def toggle_state(state):
    '''
    Toggles the state, returning True if the state was False, and False if it was True
    '''
    if state: return False
    else: return True

# def update_chunk(old_chunk, new_chunk):
#     '''
#     Will update both chunks and overlap as the slider is adjusted.
#     '''
#     logger.info(f"Updating")
#     return chunk


def update_drop_down(choices: list, request: gr.Request, 
                     choice: str | None = None):
    '''
    Updates a drop down menu. Useful for rule systems, documents, and model choices
    '''
    set_current_user(request.username)
    if choices and choice:
        # logger.info(f"Updating drop down | {choices} | {choice}") 
        return gr.Dropdown(choices = choices, value = choice)
    elif choices:
        # logger.info(f"Updating drop down | {choices}") 
        return gr.Dropdown(choices = choices)
    else:
        # logger.info(f"Updating drop down | []") 
        return gr.Dropdown(choices = None)
    

def update_login(user: str, password: str, request: gr.Request):
    '''
    Updates the user or creates a new one.
    '''
    users: dict = user_logins(return_dict=True)
    update_users = {"users": {user: password}}
    updated_users = {"users": users["users"] | update_users["users"]}
    with open(cwd / "Settings" / "Users.toml", "w") as file:
        toml.dump(updated_users, file)

    return gr.Textbox(value = user), gr.Textbox(value = "")

    


def update_number(value):
    '''
    Updates the number in the number box
    '''
    return gr.Number(value = value)


def update_slider(value, percent = 1):
    '''
    Updates the slider value
    '''
    if isinstance(value, int): return gr.Slider(value = int(percent * value))
    if isinstance(value, float): return gr.Slider(value = float(percent * value))
    return gr.Slider(value = percent * value)


def update_system_prompt(new_prompt, current):
    '''
    Only updates if the new prompt is not empty.
    Returns the updated state and refreshes the display.
    '''
    if new_prompt.strip():
        return new_prompt.strip(), new_prompt.strip(), ""
    return current, current, ""


def update_textarea(message):
    '''
    '''
    return gr.TextArea(value = message)

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


def user_submit(user_msg, chat_history, request: gr.Request,
                *args, **kwargs):
    '''
    '''
    set_current_user(request.username)
    user_msg = _extract_content(user_msg)  # normalize incoming content
    chat_history.append({"role": "user", "content": user_msg})
    chat_history.append({"role": "assistant", "content": ""})
    return "", chat_history


def user_logins(return_dict: bool = False) -> list | dict:
    '''
    This is not the ideal way to do this, but this is also meant as a simple project for just a few people.
    '''
    with open(cwd / "Settings" / "Users.toml", "r") as file:
        logins: dict = toml.load(file)

    if return_dict: users: dict = logins

    else: users: list = [(user, login) for user, login in logins["users"].items()]

    return users

# if __name__ in "__main__":
#     setup_logs(pathlib.Path(basename(__file__)).stem)

print("Finished reading Functions file")
logger.info(f"Finished reading Gradio functions file")

