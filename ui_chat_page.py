import gradio as gr
import ollama


# The overall system content, how the system reacts.
SYSTEM_CONTENT = "You are a DM for tabletop RPGs named Technomancer and are friendly. Assume the user already knows a lot of the terminology. You are not meant to generate new campaign ideas, rules, but are meant to help reference rules, tables, pages, NPCs, and the like."
SYSTEM_MODEL = "phi4"


def generate_response(message, history, system_content,
                      model = "phi4",
                      *args, **kwargs):
    '''
    This generates a response for the chatbot
    '''
    response = ""

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



# ----------------------- Layout of the interface ----------------------- #
with gr.Blocks() as page:
    system_content_state = gr.State(value = SYSTEM_CONTENT)
    system_model = gr.State(value = SYSTEM_MODEL)
    with gr.Row():
        chatbot = gr.Chatbot(type = "messages")  # have to use messages as any (or most?) other option will be deprecated. This requires the use of dictionary entries
                                                 # {"role": "user", "content": message} this is the input query
                                                 # {"role": "assistant", "content": ""} this is the bots response
                                                 # {"role": "system", "content": ""} this sets the tone of the bot. Some tones wont work.

    with gr.Row(variant = "compact"):
        with gr.Column(scale = 12):
            msg = gr.Textbox(show_label = True, label = "Input for Technomancer", submit_btn = True)
        with gr.Column(scale = 1):
            with gr.Row():
                save_btn = gr.Button(value = "Save Chat (not implemented)")
            with gr.Row():
                clear_btn = gr.ClearButton([msg, chatbot], value = "Clear Chat")

    with gr.Row():
        with gr.Column():
            upload_btn = gr.Button(value = "Upload Document")
        with gr.Column():
            blank_btn = gr.Button(value = "Nothing Yet")
        with gr.Column():
            logout_btn = gr.Button(value = "Return Home", link = "http://127.0.0.1:7860/")

    with gr.Row():
        with gr.Accordion(label = "Advanced Features", open = False):
            gr.HTML("<p>Note, changing some of these features mid conversation might cause confusion/hallucinations within the chatbot.</p>")
            
            active_prompt_display = gr.Textbox(label = "Current System State", value = SYSTEM_CONTENT, visible = False, interactive = False)
            change_sys_prompt = gr.Textbox(show_label = True, placeholder = SYSTEM_CONTENT, label = "Change System Prompt: Press enter to update", submit_btn = True)
            model_choice = gr.Dropdown(choices = ["phi4"], label = "Model Choice", info = "Only one model choice right now, function does not work.")

    # ----- Event handlers for the chatbot ----- #
    def user_submit(user_msg, chat_history, *args, **kwargs):
        '''
        This formats the user input to the proper message type.
        *args and **kwargs are not really used here, just there *in case*
        '''
        chat_history.append({"role": "user", "content": user_msg})
        chat_history.append({"role": "assistant", "content": ""})
        return "", chat_history
    
    def technomancer(chat_history, system_content, *args, **kwargs):
        '''
        This yields the chatbots response. Again, *args and **kwargs are not really needed, but are here *in case*
        '''
        user_msg = chat_history[-2]["content"]  # grab the content of the users message

        for response in generate_response(user_msg, chat_history[:-2], system_content, *args, **kwargs):
            chat_history[-1]["content"] = response  # now the last item is the assistant placeholder
            yield chat_history

    def update_system_prompt(new_prompt, current):
        '''
        Only updates if the new prompt is not empty.
        Returns the updated state and refreshes the display.
        '''
        if new_prompt.strip():
            return new_prompt.strip(), new_prompt.strip(), ""
        return current, current, ""


    msg.submit(user_submit, [msg, chatbot], [msg, chatbot], queue = False).then(technomancer, [chatbot, system_content_state], chatbot)

    change_sys_prompt.submit(update_system_prompt, [change_sys_prompt, system_content_state], [system_content_state, active_prompt_display, change_sys_prompt])


if __name__ in "__main__":
    page.launch()