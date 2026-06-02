import logging
logger = logging.getLogger(__name__)

import gradio as gr

from __rag_pipeline import find_documents

from __tech_fn import user_submit, technomancer_response, update_system_prompt, update_drop_down, update_textbox




def create_chat(embed_models, lang_models, rule_systems):
    '''
    '''
    documents_listed = gr.State([])
    
    with gr.Blocks() as chat:
        with gr.Row():
            chatbot = gr.Chatbot()

        with gr.Row(equal_height = True):
            with gr.Column(scale = 12):
                msg_box = gr.Textbox(show_label = True, label = "Enter Message", submit_btn = True)
            
            with gr.Column(scale = 1):
            
                with gr.Row():
                    save_btn = gr.Button(value = "Export Chat to DBoH (not implimented yet)")
            
                with gr.Row():
                    clear_btn = gr.ClearButton([msg_box, chatbot], value = "Clear Chat")
            
                with gr.Row():
                    stop_btn = gr.Button(value = "Stop")

        with gr.Accordion(label = "Options", open = False) as options_acc:                
            with gr.Accordion(label = "Documents", open = False) as docs_acc:
                gr.Markdown("To add or remove a document, go to Upload Tab.")
            
                with gr.Row(equal_height = True):
                    use_rag_check = gr.Checkbox(label = "Look at Rulebooks & Notes", value = False, info = "When enabled, Technomancer will search the available rule systems for answers.", visible = False)

                    collection_choice_dd = gr.Dropdown(label = "Rule System", choices = [], interactive = True)

                    available_documents_dd = gr.Dropdown(label = "Available Documents in Selected Rule System for reference.", choices = [], interactive = True)

            with gr.Accordion(label = "Model Choice", open = False) as adv_feat_acc:
                gr.Markdown("Note, changing this mid conversation might cause confusion/hallucinations within the chatbot.")
                # Add logging here to check if the model has been changed during conversation.
                model_choice = gr.Dropdown(label = "Available language models", choices = [], info = "This may report both Language models and Embedding models: please know which one is which when selecting here! You want the Language model.", interactive = True)
                # Add logging here to both of these: one to check if the embedding model has been changed, another to check if the database has been changed.
                embedding_choice = gr.Dropdown(info = "To be implemented", label = "Embedding Model Choices", choices = [], interactive = True)
                database_choice = gr.Dropdown(info = "This is linked to the embedding model choice. Once fully implemented, this will tell the user what databases are available with the chosen embedding model.", label = "Choices of Database", choices = [], interactive = True)

        chat_response = msg_box.submit(user_submit, [msg_box, chatbot], [msg_box, chatbot], queue = False).then(technomancer_response, [chatbot, model_choice, embedding_choice, collection_choice_dd, use_rag_check], chatbot)
        
        collection_choice_dd.select(fn = find_documents, inputs = [collection_choice_dd], outputs = [documents_listed]).then(fn = update_drop_down, inputs = [documents_listed], outputs = [available_documents_dd])
        
        stop_btn.click(fn = None, inputs = None, outputs = None, cancels = [chat_response])

    return chat, {"model_choice": model_choice, "collection_choice": collection_choice_dd, "embedding_choice": embedding_choice, "chatbot": chatbot}



if __name__ in "__main__":
    TECH_CHAT, _ = create_chat(["Embedding Choice"], ["Language Choice"], ["AD&D", "Shadowrun", "Rifts"])

print("Rendered Chat Tab")
# Log output that it has been uploaded