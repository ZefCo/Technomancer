import gradio as gr
from __tech_fn import user_submit, technomancer_response, update_system_prompt, update_drop_down, update_textbox
from __rag_pipeline import find_documents




def create_chat(models, rule_systems, system_content, embeddings):
    '''
    '''
    documents_listed = gr.State([])
    no_sys_prompt = gr.State(value = "")
    
    with gr.Blocks() as chat:
        with gr.Row():
            chatbot = gr.Chatbot()

        with gr.Row(equal_height = True):
            with gr.Column(scale = 12):
                msg = gr.Textbox(show_label = True, label = "Enter Message", submit_btn = True)
            
            with gr.Column(scale = 1):
            
                with gr.Row():
                    save_btn = gr.Button(value = "Save Chat to DBoH (not implimented yet)")
            
                with gr.Row():
                    clear_tn = gr.ClearButton([msg, chatbot], value = "Clear Chat")
            
                with gr.Row():
                    stop_btn = gr.Button(value = "Stop")

        with gr.Accordion(label = "Options", open = False) as ops:                
            with gr.Accordion(label = "Documents", open = False) as docs:
                gr.Markdown("To add or remove a document, go to Upload Tab.")
                # with gr.Row():
                    # with gr.Column():
                        # available_collections = gr.Dropdown(label = "Available Rule Systems", choices=[], interactive = True)
                    # with gr.Column():
                # Manual RAG control
            
                with gr.Row(equal_height = True):
                    use_rag_toggle = gr.Checkbox(label = "Look at Rulebooks & Notes", value = False, info = "When enabled, Technomancer will search the available rule systems for answers.", visible = False)

                    collection_choice = gr.Dropdown(label = "Rule System", choices = [], interactive = True)

                    available_documents = gr.Dropdown(label = "Available Documents in Selected Rule System for reference.", choices = [], interactive = True)

            # with gr.Accordion(label = "Advanced Features", open = False) as adv_feat:
            with gr.Accordion(label = "Model Choice", open = False) as adv_feat:
                # with gr.Row():
                # gr.Markdown("Note, changing some of these features mid conversation might cause confusion/hallucinations within the chatbot.")
                gr.Markdown("Note, changing this mid conversation might cause confusion/hallucinations within the chatbot.")
            
                # with gr.Row():
                    # active_prompt_display = gr.Textbox(label = "Current System State", value = system_content, visible = False, interactive = False)
                    # change_sys_prompt = gr.Textbox(show_label = True, placeholder = None, label = "Change System Prompt: Press enter to update", submit_btn = True)  # Do I really need the ability to change the system prompt?    
                    # clear_sys_prompt = gr.Button(value = "Clear system prompt")  # Do I really need this? Probably not.
    
                model_choice = gr.Dropdown(label = "Available language models", choices = [], info = "This may report both Language models and Embedding models: please know which one is which when selecting here! You want the Language model.", interactive = True)
                embedding_choice = gr.Dropdown(info = "To be implemented", label = "Embedding Model Choices", choices = [], interactive = True)
                database_choice = gr.Dropdown(info = "This is linked to the embedding model choice. Once fully implemented, this will tell the user what databases are available with the chosen embedding model.", label = "Choices of Database", choices = [], interactive = True)

        # Need to trigger the updates when the according is expanded or collapsed. That way we can easily pass in the state variables.
        # adv_feat.expand(fn = update_textbox, inputs = [system_content], outputs = [change_sys_prompt]).then(fn = update_drop_down, inputs = [models], outputs = [model_choice]).then(fn = update_drop_down, inputs = [rule_systems], outputs = [collection_choice])
        # adv_feat.collapse(fn = update_textbox, inputs = [system_content], outputs = [change_sys_prompt]).then(fn = update_drop_down, inputs = [models], outputs = [model_choice]).then(fn = update_drop_down, inputs = [rule_systems], outputs = [collection_choice])
        
        # docs.expand(fn = update_drop_down, inputs = [rule_systems], outputs = [collection_choice])

        # clear_sys_prompt.click(update_system_prompt, [no_sys_prompt, system_content], [system_content, active_prompt_display, change_sys_prompt]).then(update_textbox, [system_content], [change_sys_prompt])

        chat_response = msg.submit(user_submit, [msg, chatbot], [msg, chatbot], queue = False).then(technomancer_response, [chatbot, system_content, model_choice, embedding_choice, collection_choice, use_rag_toggle], chatbot)
        
        # change_sys_prompt.submit(update_system_prompt, [change_sys_prompt, system_content], [system_content, active_prompt_display, change_sys_prompt]).then(update_textbox, [system_content], [change_sys_prompt])
        
        collection_choice.select(fn = find_documents, inputs = [collection_choice], outputs = [documents_listed]).then(fn = update_drop_down, inputs = [documents_listed], outputs = [available_documents])
        
        stop_btn.click(fn = None, inputs = None, outputs = None, cancels = [chat_response])

        # chat.load(fn = update_textbox, inputs = [system_content], outputs = [change_sys_prompt]).then(fn = update_drop_down, inputs = [models], outputs = [model_choice]).then(fn = update_drop_down, inputs = [rule_systems], outputs = [collection_choice])
        # chat.load(fn = update_drop_down, inputs = [models], outputs = [model_choice]).then(fn = update_drop_down, inputs = [rule_systems], outputs = [collection_choice]).then(fn = update_drop_down, inputs = [embeddings], outputs = [embedding_choice])

    return chat, {"model_choice": model_choice, "collection_choice": collection_choice, "embedding_choice": embedding_choice}

print("Rendered Chat Page")


# if __name__ in "__main__":
#     chat.launch()