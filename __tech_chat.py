# from datetime import datetime
# import pathlib

import logging

from os.path import basename
from __log_fn import setup_logs

import gradio as gr

from __rag_pipeline import find_documents

from __tech_fn import change_state, technomancer_response, update_system_prompt, update_drop_down, update_textbox, user_submit

logger = logging.getLogger(__name__)


def create_chat(embed_models, lang_models, rule_systems):
    '''
    '''
    documents_listed_s= gr.State([])
    embed_name_s = gr.State(value = "Embedding Model")
    false_state_s = gr.State(value = False)
    lang_name_s = gr.State(value = "Language Model")
    previous_state_lang_s = gr.State(value = None)
    previous_state_embed_s = gr.State(value = None)
    true_state_s = gr.State(value = True)

    
    with gr.Blocks() as chat:
        with gr.Row():
            chatbot = gr.Chatbot(buttons = ["copy_all"])

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

                    available_rule_systems_dd = gr.Dropdown(label = "Rule System", choices = [], interactive = True)

                    available_documents_dd = gr.Dropdown(label = "Available Documents in Selected Rule System for reference.", choices = [], interactive = True)

            with gr.Accordion(label = "Model Choice", open = False) as adv_feat_acc:
                gr.Markdown("Note, changing this mid conversation might cause confusion/hallucinations within the chatbot.")
                model_choice_dd = gr.Dropdown(label = "Available language models", choices = [], info = "This may report both Language models and Embedding models: please know which one is which when selecting here! You want the Language model.", interactive = True)
                embedding_choice_dd = gr.Dropdown(info = "To be implemented", label = "Embedding Model Choices", choices = [], interactive = False)
                database_choice_dd = gr.Dropdown(info = "This is linked to the embedding model choice. Once fully implemented, this will tell the user what databases are available with the chosen embedding model.", label = "Choices of Database", choices = [], interactive = True)

        chat_response = msg_box.submit(user_submit, [msg_box, chatbot], [msg_box, chatbot], queue = False).then(technomancer_response, [chatbot, model_choice_dd, embedding_choice_dd, available_rule_systems_dd, use_rag_check], chatbot)
        
        available_rule_systems_dd.select(fn = find_documents, inputs = [available_rule_systems_dd], outputs = [documents_listed_s]).then(fn = update_drop_down, inputs = [documents_listed_s], outputs = [available_documents_dd])
        
        stop_btn.click(fn = None, inputs = None, outputs = None, cancels = [chat_response])

        model_choice_dd.select(fn = change_state, inputs = [model_choice_dd, previous_state_lang_s, true_state_s, lang_name_s], outputs = [previous_state_lang_s])

        # Add logging ability for what database choices show up too
        embedding_choice_dd.select(fn = change_state, inputs = [embedding_choice_dd, previous_state_embed_s, true_state_s, embed_name_s], outputs = [previous_state_embed_s])

    return chat, {"model_choice": model_choice_dd, "collection_choice": available_rule_systems_dd, "embedding_choice": embedding_choice_dd, "chatbot": chatbot}



# if __name__ in "__main__":
#     setup_logs(pathlib.Path(basename(__file__)).stem)


if __name__ in "__main__":
    from datetime import datetime
    import pathlib

    cwd = pathlib.Path.cwd()
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")    

    log_dir = cwd / "Logs"
    log_dir.mkdir(parents = True, exist_ok = True)
    
    logger = setup_logs(__name__, log_dir / f"{pathlib.Path(basename(__file__)).stem}_{timestamp}.log")

    TECH_CHAT, _ = create_chat(["Embedding Choice"], ["Language Choice"], ["AD&D", "Shadowrun", "Rifts"])
else:
    logger = logging.getLogger(__name__)
    print("Rendered Chat Tab")
    logger.info("Rendered Chat Tab")
