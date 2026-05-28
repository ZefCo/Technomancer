import gradio as gr
import ollama
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_classic.chains import Re

from ui_functions import generate_response, load_settings

SETTINGS = load_settings()
RULE_SYSTEMS = SETTINGS["rule_systems"]  # Note this will have to also update on refresh to see if there are any new rule systems. This links back to the idea of global, system and browser states.
# MODELS = SETTINGS["models"]

# The overall system content, how the system reacts.
SYSTEM_CONTENT = "You are a DM for tabletop RPGs named Technomancer and are friendly. Assume the user already knows a lot of the terminology. You are not meant to generate new campaign ideas, rules, but are meant to help reference rules, tables, pages, NPCs, and the like."
SYSTEM_MODELS = ["phi4"]
print("Finished loading Chat Page")




# ----------------------- Layout of the interface ----------------------- #
with gr.Blocks() as page:
    system_content_state = gr.State(value = SYSTEM_CONTENT)
    system_model = gr.State(value = SYSTEM_MODELS)
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

    # with gr.Row():
    #     with gr.Column():
    #         upload_btn = gr.Button(value = "Upload Document")
    #     with gr.Column():
    #         blank_btn = gr.Button(value = "Nothing Yet")
    #     # with gr.Column():
    #         # logout_btn = gr.Button(value = "Return Home", link = "http://127.0.0.1:7860/")
    with gr.Row():
        with gr.Accordion(label = "Documents", open = False):
            gr.HTML("To add or remove a document, go to the Upload page.")
            with gr.Row():
                with gr.Column():
                    available_collections = gr.Dropdown(choices = ["Collection 1", "Collection 2", "Collection 3"], info = "Choice of different collections", interactive = True)
                with gr.Column():
                    available_documents = gr.Dropdown(choices = ["Doc 1", "Doc 2", "Doc 3"], info = "All available documents in that collection", interactive = True)

    with gr.Row():
        with gr.Accordion(label = "Advanced Features", open = False):
            gr.HTML("<p>Note, changing some of these features mid conversation might cause confusion/hallucinations within the chatbot.</p>")
            
            active_prompt_display = gr.Textbox(label = "Current System State", value = SYSTEM_CONTENT, visible = False, interactive = False)
            change_sys_prompt = gr.Textbox(show_label = True, placeholder = SYSTEM_CONTENT, label = "Change System Prompt: Press enter to update", submit_btn = True)
            model_choice = gr.Dropdown(choices = SYSTEM_MODELS, value = SYSTEM_MODELS[0], label = "Model Choice", info = "This will report both Language models and Embedding models: please know which one is which when selecting here! You want the Language model.", interactive = True)
            # Manual RAG control
            with gr.Row():
                use_rag_toggle = gr.Checkbox(label = "Look at Rulebooks & Notes", value = False, info = "When enabled, Technomancer will search the available rule systems for answers.")
                collection_choice = gr.Dropdown(choices = RULE_SYSTEMS, value = RULE_SYSTEMS[0] if RULE_SYSTEMS else None, label = "Rule System", interactive = True)



    # ----- Event handlers for the chatbot ----- # See if these can be moved to the UI functions.
    def user_submit(user_msg, chat_history, *args, **kwargs):
        '''
        This formats the user input to the proper message type.
        *args and **kwargs are not really used here, just there *in case*
        '''
        chat_history.append({"role": "user", "content": user_msg})
        chat_history.append({"role": "assistant", "content": ""})
        return "", chat_history
    
    def technomancer(chat_history, system_content, model, *args, **kwargs):
        '''
        This yields the chatbots response. Again, *args and **kwargs are not really needed, but are here *in case*
        '''
        user_msg = chat_history[-2]["content"]  # grab the content of the users message

        for response in generate_response(user_msg, chat_history[:-2], system_content, model, *args, **kwargs):
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


    msg.submit(user_submit, [msg, chatbot], [msg, chatbot], queue = False).then(technomancer, [chatbot, system_content_state, model_choice, collection_choice, use_rag_toggle], chatbot)

    change_sys_prompt.submit(update_system_prompt, [change_sys_prompt, system_content_state], [system_content_state, active_prompt_display, change_sys_prompt])


if __name__ in "__main__":
    page.launch()