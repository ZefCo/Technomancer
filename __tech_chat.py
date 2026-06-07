# from datetime import datetime
# import pathlib

import logging

from os.path import basename
from __log_fn import setup_logs

import gradio as gr

from __states import (avatars_state, 
                      documents_list_state, 
                      embed_model_state, embed_models_list_state, 
                      lang_model_state, lang_models_list_state, 
                      name_embed_state, name_lang_state, 
                      rule_system_state, 
                      true_state)

from __rag_pipeline import find_documents

from __tech_fn import change_state, technomancer_response, update_drop_down, update_textbox, user_submit, list_length


def create_chat():
    '''
    '''
    
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

        with gr.Row():
            lang_textbox = gr.Textbox(label = "Langage Model Being Used", value = "")
            embed_textbox = gr.Textbox(label = "Embedding Model Being Used", value = "")

        with gr.Accordion(label = "Options", open = False) as options_acc:                
            with gr.Accordion(label = "Documents", open = False) as docs_acc:
                with gr.Row():
                    metadata_tags_dd = gr.Dropdown(choices = [], label = "Metadata Tags: select to add additional filtering when querying responses. This does not allow custom inputs - custom tags are to be input in the Upload Tab", interactive = True, multiselect = True)

                gr.Markdown("To add or remove a document, go to Upload Tab. This is for references.")
            
                with gr.Row(equal_height = True):
                    # use_rag_check = gr.Checkbox(label = "Look at Rulebooks & Notes", value = False, info = "When enabled, Technomancer will search the available rule systems for answers.", visible = False)

                    available_rule_systems_dd = gr.Dropdown(label = "Rule System", choices = [], interactive = True, scale = 10)
                    available_documents_textbox = gr.Textbox(label = "Docs Found", value = "", scale = 1)

                    available_documents_dd = gr.Dropdown(label = "Available Documents in Selected Rule System for reference.", choices = [], interactive = True, scale = 15)

            with gr.Accordion(label = "Model Choice", open = False) as adv_feat_acc:
                gr.Markdown("Note, changing this mid conversation might cause confusion/hallucinations within the chatbot.")
                lang_model_choice_dd = gr.Dropdown(label = "Available language models", choices = [], info = "This may report both Language models and Embedding models: please know which one is which when selecting here! You want the Language model.", interactive = True)
                embedding_choice_dd = gr.Dropdown(info = "To be implemented", label = "Embedding Model Choices", choices = [], interactive = True)
                database_choice_dd = gr.Dropdown(info = "This is linked to the embedding model choice. Once fully implemented, this will tell the user what databases are available with the chosen embedding model.", label = "Choices of Database", choices = [], interactive = True)

        chat_response = msg_box.submit(user_submit, [msg_box, chatbot], [msg_box, chatbot], queue = False).then(fn = technomancer_response, inputs = [chatbot, lang_model_choice_dd, embedding_choice_dd, available_rule_systems_dd, metadata_tags_dd], outputs = [chatbot])
        
        available_rule_systems_dd.select(fn = find_documents, inputs = [available_rule_systems_dd], outputs = [documents_list_state]).then(fn = update_drop_down, inputs = [documents_list_state], outputs = [available_documents_dd]).then(fn = list_length, inputs = [documents_list_state], outputs = [available_documents_textbox])
        
        stop_btn.click(fn = None, inputs = None, outputs = None, cancels = [chat_response])

        lang_model_choice_dd.select(fn = change_state, inputs = [lang_model_choice_dd, lang_model_state, true_state, name_lang_state], outputs = [lang_model_state]).then(fn = update_textbox, inputs = [lang_model_state], outputs = [lang_textbox])

        embedding_choice_dd.select(fn = change_state, inputs = [embedding_choice_dd, embed_model_state, true_state, name_embed_state], outputs = [embed_model_state]).then(fn = update_textbox, inputs = [embed_model_state], outputs = [embed_textbox])

    return chat, {"embed_models_dd": embedding_choice_dd, "embed_textbox": embed_textbox, "lang_models_dd": lang_model_choice_dd, "lang_textbox": lang_textbox, "metadata_tags_dd": metadata_tags_dd, "rule_systems_dd": available_rule_systems_dd}


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
    logger.info("Rendered Chat Tab @ (time to be implemented)")
