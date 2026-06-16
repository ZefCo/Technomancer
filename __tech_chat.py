# from datetime import datetime
# import pathlib

import logging

from os.path import basename
from __log_fn import setup_logs

import gradio as gr

from __states import (avatars_state, 
                      documents_list_state, 
                      embed_model_state, embed_models_list_state, 
                      k_state,
                      lang_model_state, lang_models_list_state, 
                      name_embed_state, name_k_state, name_lang_state, name_threshold_state,
                      prefix_state,
                      rule_system_state, 
                      threshold_state, true_state,
                      user_state)

from __rag_pipeline import find_documents, find_document

from __tech_fn import change_state, technomancer_response, update_drop_down, update_textbox, user_submit, list_length, enable_prefix, toggle_state


def create_chat():
    '''
    '''
    
    with gr.Blocks() as chat:
        # user_info = {"user_name": user_state}
        # logger = logger.LoggerAdapter(logger, user_info)
        # with gr.Sidebar(position = "left", open = False) as sidebar:
        #     logout = gr.Button("Logout", link = "/logout")

        with gr.Row():
            try:
                chatbot = gr.Chatbot(buttons = ["copy_all"])
            except Exception as e:
                logger.critical(f"{user_state} | Error at initializing chatbot | Commonly caused by using Gradio version < 6 | {type(e)} | {e}")
                print("Critical Error, check logs for error - possible cause: using Gradio < Gradio 6")
                import sys
                sys.exit()

        with gr.Row(equal_height = True):
            with gr.Column(scale = 12):
                msg_box = gr.Textbox(show_label = True, label = "Enter Message (shift enter to send)", submit_btn = True, lines = 5)
            
            with gr.Column(scale = 1):
            
                with gr.Row():
                    save_btn = gr.Button(value = "Export Chat to DBoH (not implimented yet)")
            
                with gr.Row():
                    clear_btn = gr.ClearButton([msg_box, chatbot], value = "Clear Chat")
            
                with gr.Row():
                    stop_btn = gr.Button(value = "Stop")

        with gr.Row(equal_height = True):
            lang_textbox = gr.Textbox(label = "Langage Model Being Used", value = "", scale = 10)
            embed_textbox = gr.Textbox(label = "Embedding Model Being Used", value = "", scale = 10)
            with gr.Column():
                prefix_check = gr.Checkbox(label = "Use Prefix", visible = False, scale = 1, interactive = True)
                prefix_box = gr.Textbox(value = "Allows for Prefixes to be used on query - not yet turned on", visible=False)

        with gr.Accordion(label = "Options", open = False) as options_acc:                
            with gr.Accordion(label = "Documents", open = False) as docs_acc:
                with gr.Row():
                    metadata_tags_dd = gr.Dropdown(choices = [], label = "Metadata Tags: select to add additional filtering when querying responses. This does not allow custom inputs - custom tags are to be input in the Upload Tab", interactive = True, multiselect = True)

                gr.Markdown("To add or remove a document, go to Upload Tab. This is for references.")
            
                with gr.Row(equal_height = True):
                    # use_rag_check = gr.Checkbox(label = "Look at Rulebooks & Notes", value = False, info = "When enabled, Technomancer will search the available rule systems for answers.", visible = False)

                    available_rule_systems_dd = gr.Dropdown(label = "Rule System", choices = [], interactive = True, scale = 10)
                    available_documents_textbox = gr.Textbox(label = "Docs Found", value = "", scale = 1)

                    available_documents_dd = gr.Dropdown(label = "Available Documents in Selected Rule System (for reference).", choices = [], interactive = True, scale = 15)

            with gr.Accordion(label = "Model & Query Choices", open = False) as adv_feat_acc:
                gr.Markdown("Note, changing this mid conversation might cause confusion/hallucinations within the chatbot.")
                with gr.Column(variant = "panel"):
                    with gr.Row():
                        lang_model_choice_dd = gr.Dropdown(label = "Available language models", choices = [], info = "This may report both Language models and Embedding models: please know which one is which when selecting here! You want the Language model.", interactive = True)
                        embedding_choice_dd = gr.Dropdown(label = "Embedding Model Choices", choices = [], interactive = True)
                
                # database_choice_dd = gr.Dropdown(info = "Once fully implemented, this will tell the user what databases are available.", label = "Choices of Database", choices = [], interactive = False)

                with gr.Column(variant = "panel"):
                    with gr.Row():
                        k_num = gr.Number(label = "Top K Queries returned", value = None, scale = 3)
                        threshold = gr.Slider(label="Cosine Similarity Threshold", minimum=0, maximum=2, step=0.001, scale = 10, info="0 -> totally similar, 2 -> complete opposite")
                # # update_khold = gr.Button(value = "Update K and Threshold", scale = 3)


        available_rule_systems_dd.select(fn = find_documents, inputs = [available_rule_systems_dd], outputs = [documents_list_state]).then(fn = update_drop_down, inputs = [documents_list_state], outputs = [available_documents_dd]).then(fn = list_length, inputs = [documents_list_state], outputs = [available_documents_textbox])
        available_documents_dd.select(fn = find_document, inputs = [available_documents_dd, available_rule_systems_dd])
        
        chat_response = msg_box.submit(user_submit, [msg_box, chatbot], [msg_box, chatbot], queue = False).then(fn = technomancer_response, inputs = [chatbot, lang_model_choice_dd, embedding_choice_dd, available_rule_systems_dd, metadata_tags_dd, k_state, threshold_state], outputs = [chatbot])
        
        embedding_choice_dd.select(fn = change_state, inputs = [embedding_choice_dd, embed_model_state, true_state, name_embed_state], outputs = [embed_model_state]).then(fn = update_textbox, inputs = [embed_model_state], outputs = [embed_textbox]).then(fn = enable_prefix, inputs = [embed_model_state], outputs = [prefix_check, prefix_box])

        k_num.change(fn = change_state, inputs = [k_num, k_state, true_state, name_k_state], outputs = [k_state])

        lang_model_choice_dd.select(fn = change_state, inputs = [lang_model_choice_dd, lang_model_state, true_state, name_lang_state], outputs = [lang_model_state]).then(fn = update_textbox, inputs = [lang_model_state], outputs = [lang_textbox])

        prefix_check.select(fn = toggle_state, inputs = [prefix_state], outputs = [prefix_state])

        stop_btn.click(fn = None, inputs = None, outputs = None, cancels = [chat_response])

        threshold.change(fn = change_state, inputs = [threshold, threshold_state, true_state, name_threshold_state], outputs = [threshold_state])

    return chat, {"embed_models_dd": embedding_choice_dd, "embed_textbox": embed_textbox, "k": k_num, "lang_models_dd": lang_model_choice_dd, "lang_textbox": lang_textbox, "metadata_tags_dd": metadata_tags_dd, "rule_systems_dd": available_rule_systems_dd, "threshold": threshold}
    # return chat, {"embed_models_dd": embedding_choice_dd, "embed_textbox": embed_textbox, "lang_models_dd": lang_model_choice_dd, "lang_textbox": lang_textbox, "metadata_tags_dd": metadata_tags_dd, "rule_systems_dd": available_rule_systems_dd, "threshold": threshold}
    # return chat, {"embed_models_dd": embedding_choice_dd, "embed_textbox": embed_textbox, "lang_models_dd": lang_model_choice_dd, "lang_textbox": lang_textbox, "metadata_tags_dd": metadata_tags_dd, "rule_systems_dd": available_rule_systems_dd}


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
