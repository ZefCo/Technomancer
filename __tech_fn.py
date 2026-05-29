# for functions
import subprocess
import ollama
import gradio as gr
import re



def find_models():
    '''
    Finds all the models that are installed on the computer. Meant to be run when the server starts.
    '''
    result = subprocess.run(['ollama', 'list'], capture_output=True, text=True)
    lines = result.stdout.strip().splitlines()[1:]  # Skip header line
    models = [line.split()[0] for line in lines]  # Get model names from the first column
    return models


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




def generate_response(message, history, system_content, model,
                      collection = None, use_rag = False,
                      *args, **kwargs):
    '''
    This will route between RAG and the Ollama chat depending on the if a collection is used or not, which hopefully is triggered properly in the context of the message.
    '''
    if use_rag and collection:
        pass
        from __rag_pipeline import query_rag
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



def technomancer_response(chat_history, system_content, model, *args, **kwargs):
    '''
    This yields the chatbots response. Again, *args and **kwargs are not really needed, but are here *in case*
    '''
    user_msg = chat_history[-2]["content"]  # grab the content of the users message

    for response in generate_response(user_msg, chat_history[:-2], system_content, model, *args, **kwargs):
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


def update_state_list(states, state):
    '''
    Updates a state list with a new stat added.
    '''
    if state and state not in tuple(states):
        states.append(state)
        states.sort()
    return states


def user_submit(user_msg, chat_history, *args, **kwargs):
    '''
    This formats the user input to the proper message type.
    *args and **kwargs are not really used here, just there *in case*
    '''
    chat_history.append({"role": "user", "content": user_msg})
    chat_history.append({"role": "assistant", "content": ""})
    return "", chat_history
