import gradio as gr
from __tech_fn import user_submit, technomancer_response, update_system_prompt, update_drop_down, update_textbox
from __rag_pipeline import find_documents




def create_chat(system_content, rule_systems, models):
    '''
    '''
    documents_listed = gr.State([])
    with gr.Blocks() as chat:
        with gr.Row():
            chatbot = gr.Chatbot(type = "messages")

        with gr.Row():
            with gr.Column(scale = 12):
                msg = gr.Textbox(show_label = True, label = "Msg for Technomancer", submit_btn = True)
            with gr.Column(scale = 1):
                with gr.Row():
                    save_btn = gr.Button(value = "Save Chat to DBoH (not implimented yet)")
                with gr.Row():
                    clear_tn = gr.ClearButton([msg, chatbot], value = "Clear Chat")

        with gr.Row():
            with gr.Accordion(label = "Documents", open = False) as docs:
                gr.Markdown("To add or remove a document, go to Upload Tab.")
                # with gr.Row():
                    # with gr.Column():
                        # available_collections = gr.Dropdown(label = "Available Rule Systems", choices=[], interactive = True)
                    # with gr.Column():
                # Manual RAG control
                with gr.Row():
                    use_rag_toggle = gr.Checkbox(label = "Look at Rulebooks & Notes", value = False, info = "When enabled, Technomancer will search the available rule systems for answers.")
                    collection_choice = gr.Dropdown(choices = [], label = "Rule System", interactive = True)
                    available_documents = gr.Dropdown(label = "Available Documents in Selected Rule System. Selection does nothing, just for reference", choices = [], interactive = True)

        with gr.Row():
            with gr.Accordion(label = "Advanced Features", open = False) as adv_feat:
                gr.Markdown("Note, changing some of these features mid conversation might cause confusion/hallucinations within the chatbot.")
                active_prompt_display = gr.Textbox(label = "Current System State", value = system_content, visible = False, interactive = False)
                change_sys_prompt = gr.Textbox(show_label = True, placeholder = None, label = "Change System Prompt: Press enter to update", submit_btn = True)
                model_choice = gr.Dropdown(choices = [], label = "Model Choice", info = "This will report both Language models and Embedding models: please know which one is which when selecting here! You want the Language model.", interactive = True)

        # Need to trigger the updates when the according is expanded or collapsed. That way we can easily pass in the state variables.
        adv_feat.expand(fn = update_textbox, inputs = [system_content], outputs = [change_sys_prompt]).then(fn = update_drop_down, inputs = [models], outputs = [model_choice]).then(fn = update_drop_down, inputs = [rule_systems], outputs = [collection_choice])
        adv_feat.collapse(fn = update_textbox, inputs = [system_content], outputs = [change_sys_prompt]).then(fn = update_drop_down, inputs = [models], outputs = [model_choice]).then(fn = update_drop_down, inputs = [rule_systems], outputs = [collection_choice])
        
        docs.expand(fn = update_drop_down, inputs = [rule_systems], outputs = [collection_choice])

        msg.submit(user_submit, [msg, chatbot], [msg, chatbot], queue = False).then(technomancer_response, [chatbot, system_content, model_choice, collection_choice, use_rag_toggle], chatbot)
        change_sys_prompt.submit(update_system_prompt, [change_sys_prompt, system_content], [system_content, active_prompt_display, change_sys_prompt]).then(update_textbox, [system_content], [change_sys_prompt])
        collection_choice.select(fn = find_documents, inputs = [collection_choice], outputs = [documents_listed]).then(fn = update_drop_down, inputs = [documents_listed], outputs = [available_documents])


    return chat

print("Rendered Chat Page")


# if __name__ in "__main__":
#     chat.launch()