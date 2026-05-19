import gradio as gr
import ollama

def format_history(msg: str, history: list[list[str, str]], system_prompt: str):
    '''
    '''
    chat_history = [{"role": "system", "content": system_prompt}]

    for query, response in history:
        chat_history.append({"role": "user", "content": query})
        chat_history.append({"role": "assistant", "content": response})

    chat_history.append({"role": "user", "content": msg})

    return chat_history


def generate_response(msg: str, history: list[list[str, str]], system_prompt: str,
                      model = "phi4"):
    '''
    '''
    chat_history = format_history(msg, history, system_prompt)

    response = ollama.chat(model = model, stream = True, messages = chat_history)

    message = ""

    for partial_resp in response:
        token = partial_resp["message"]["content"]
        
        message += token
        
        yield message


'''
Creates the interface for the chatbot.
'''
chatbot = gr.ChatInterface(
    generate_response,
    chatbot = gr.Chatbot(
        type = "messages",  # this is because gradio is moving away from using tuples for chat. Honestly not sure what to do with this, just leaving it here until it bites me in the ass.
        # avatar_images = ["SuperMarioRPGSNESCoverArtUS.jpg", "Technomancer_Devin_Larson.jpg"],  # This was playing around with avatars. Need to find something copyright free in the future.
        # height = "64vh"
    ),
    # additional_inputs = [],
    title = "Technomancer",
    submit_btn = "⬅ Send",
    # retry_btn = "Regenerate Response",  # These are no longer in the latest version. Keeping because I like the idea of them, and would like to find a way to implement them.
    # undo_btn = "↩ Undo",
    # clear_btn = "Clear History"
)

chatbot.launch()