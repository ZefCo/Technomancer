import ollama
import pathlib
cwd = pathlib.Path.cwd()
import subprocess
import yaml
import gradio as gr

print("Finished loading Functions")


def clear_button():
    '''
    '''
    return gr.update(value = None)


def load_settings():
    '''
    Imports the settings
    '''

    with open(cwd / "Settings" / "UserOptions.yaml", "r") as file:
        settings = yaml.safe_load(file)

    return settings


def write_settings_index(settings: dict, updates, index):
    '''
    Rewrites the setting file for the user.
    '''
    settings[index] = updates
    with open(cwd / "Settings" / "UserOptions.yaml", "w") as file:
        yaml.dump(settings, file)


def write_settings(settings: dict):
    '''
    Writes all settings. Useful if a lot of changes have been made.
    '''
    with open(cwd / "Settings" / "UserOptions.yaml", "w") as file:
        yaml.dump(settings, file)


def find_models():
    '''
    Finds all the models that are installed on the computer. Meant to be run when the server starts.
    '''
    result = subprocess.run(['ollama', 'list'], capture_output=True, text=True)
    lines = result.stdout.strip().splitlines()[1:]  # Skip header line
    models = [line.split()[0] for line in lines]  # Get model names from the first column
    return models


# This is the original chat response. Leaving here because I might need to reference it later.
# def generate_response_og(message, history, system_content,
#                       model,
#                       *args, **kwargs):
#     '''
#     This generates a response for the chatbot.
#     '''
#     response = ""

#     messages = [{"role": "system", "content": system_content}]

#     for item in history:
#         messages.append({"role": item["role"], "content": item["content"]})  # appends the information into the message with the proper format

#     messages.append({"role": "user", "content": message})

#     completion = ollama.chat(model = model, messages = messages, stream = True)

#     response = ""
#     for chunk in completion:
#         if "message" in chunk and "content" in chunk["message"]:
#             response += chunk["message"]["content"]
#             yield response

def generate_response(message, history, system_content, model,
                      collection = None, use_rag = False,
                      *args, **kwargs):
    '''
    This will route between RAG and the Ollama chat depending on the if a collection is used or not, which hopefully is triggered properly in the context of the message.
    '''
    if use_rag and collection:
        from RAG_pipeline import query_rag
        yield from query_rag(message, history, collection, system_content, model)
    else:
        messages = [{"role": "system", "content": system_content}]

        for item in history:
            messages.append({"role": item["role"], "content": item["content"]})  # appends the information into the message with the proper format

        messages.append({"role": "user", "content": message})

        completion = ollama.chat(model = model, messages = messages, stream = True)

        response = ""
        for chunk in completion:
            if "message" in chunk and "content" in chunk["message"]:
                response += chunk["message"]["content"]
                yield response