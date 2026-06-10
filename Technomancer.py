import pathlib
# from os.path import basename
from __log_fn import setup_logs
from datetime import datetime
import logging

cwd = pathlib.Path.cwd()
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
log_dir = cwd / "Logs"
log_dir.mkdir(parents = True, exist_ok = True)
    
# logger = logging.getLogger(__name__)
    
logger = setup_logs(log_dir / f"Technomancer__{timestamp}.log", level = logging.INFO)
# logger_debug = setup_logs(log_dir / f"Technomancer_DEBUG_{timestamp}.log")


import gradio as gr

# from __rag_pipeline import find_collections, find_documents

from __states import (avatars_state, 
                      chunk_batches_state, chunk_overlap_state, chunk_size_state, chunk_summary_state,
                      db_paths_list_state, default_message_state, documents_list_state, 
                      embed_model_state, embed_models_list_state, empty_list_state, 
                      false_state, 
                      k_state,
                      lang_model_state, lang_models_list_state, log_file_state,
                      name_chunksize_state, name_embed_state, name_k_state, name_lang_state, name_rule_state, named_chunkoverlap_state, name_tags_state, name_threshold_state, name_chunkbatch_state, name_chunksum_state,
                      percent_state, prefix_state,
                      rule_system_state, rule_systems_list_state, 
                      settings_path_tags_state, 
                      tags_list_state, true_state, threshold_state,
                      upload_status_state,
                      server_name, server_port)


import __tech_about as tech_about
from __tech_chat import create_chat
from __tech_fn import update_drop_down, update_textbox, update_slider, update_number
from __tech_upload import create_upload
from __tech_logs import create_log


def launch_technomancer():

    with gr.Blocks(title = "Technomancer vBeta") as Technomancer:
        # This is meant to better share states
        avatars_state.render()
        chunk_summary_state.render(), chunk_batches_state.render(), chunk_overlap_state.render(), chunk_size_state.render(), 
        db_paths_list_state.render(), default_message_state.render(), documents_list_state.render()
        embed_model_state.render(), embed_models_list_state.render(), empty_list_state.render()
        k_state.render()
        log_file_state.render(), lang_model_state.render(), lang_models_list_state.render()
        name_chunkbatch_state.render(), name_chunksum_state.render(), name_chunksize_state.render(), named_chunkoverlap_state.render(), name_embed_state.render(), name_k_state.render(), name_lang_state.render(), name_rule_state.render(), name_tags_state.render(), name_threshold_state.render()
        percent_state.render(), prefix_state.render()
        rule_system_state.render(), rule_systems_list_state.render()
        tags_list_state.render()
        settings_path_tags_state.render()
        threshold_state.render()
        true_state.render(), false_state.render()
        upload_status_state.render()


        # because of how many state variables I'm juggeling, I'm going to keep them alphabitized.
        with gr.Tabs():

            with gr.Tab(label = "About/Manual"):
                tech_about.about.render()
            
            with gr.Tab(label = "Technomancer Chat") as chat_tab:
                try:
                    TECH_CHAT, chat_components = create_chat()
                except Exception as e:
                    # log that chat cannot be loaded properly.
                    logger.critical(f"Something went wrong when starting Chat Tab | Error type {type(e)} | {e}")
                    raise RuntimeError("Cannot load Chat Tab")


            with gr.Tab(label = "Database of Holding") as upload_tab:
                try:
                    TECH_UPLOAD, upload_components = create_upload()
                except TypeError as e:
                    logger.critical(f"Type error when starting Upload Tab | Error type {type(e)} | {e}")
                    raise RuntimeError("Cannot load Upload Tab")
                except Exception as e:
                    # loga that upload tab cannot be loaded properly.
                    logger.critical(f"Something went wrong when starting Upload Tab | Error type {type(e)} | {e}")
                    raise RuntimeError("Cannot load Upload Tab")
                
            with gr.Tab(label = "Logs") as log_tab:
                TECH_LOG, log_components = create_log()

        chat_tab.select(fn = update_drop_down, inputs = [embed_models_list_state, embed_model_state], outputs = [chat_components["embed_models_dd"]]).then(fn = update_drop_down, inputs = [lang_models_list_state, lang_model_state], outputs = [chat_components["lang_models_dd"]]).then(fn = update_drop_down, inputs = [rule_systems_list_state], outputs = [chat_components["rule_systems_dd"]]).then(fn = update_textbox, inputs = [lang_model_state], outputs = [chat_components["lang_textbox"]]).then(fn = update_textbox, inputs = [embed_model_state], outputs = [chat_components["embed_textbox"]]).then(fn = update_drop_down, inputs = [tags_list_state], outputs = [chat_components["metadata_tags_dd"]]).then(fn = update_slider, inputs = [threshold_state], outputs = [chat_components["threshold"]]).then(fn = update_number, inputs = [k_state], outputs = [chat_components["k"]])
        upload_tab.select(fn = update_drop_down, inputs = [embed_models_list_state, embed_model_state], outputs = [upload_components["embed_models_dd"]]).then(fn = update_drop_down, inputs = [db_paths_list_state], outputs = [upload_components["list_of_db_dd"]]).then(fn = update_drop_down, inputs = [rule_systems_list_state], outputs = [upload_components["rule_systems_dd_1"]]).then(fn = update_drop_down, inputs = [rule_systems_list_state], outputs = [upload_components["rule_systems_dd_2"]]).then(fn = update_textbox, inputs = [embed_model_state], outputs = [upload_components["embed_textbox"]]).then(fn = update_drop_down, inputs = [tags_list_state], outputs = [upload_components["metadata_tags_dd"]]).then(fn = update_drop_down, inputs = [lang_models_list_state], outputs = upload_components["lang_model_sum_dd"]).then(fn = update_slider, inputs = [chunk_batches_state], outputs = [upload_components["chunk_batch"]]).then(fn = update_slider, inputs = [chunk_overlap_state], outputs = [upload_components["chunk_overlap"]]).then(fn = update_slider, inputs = [chunk_size_state], outputs = [upload_components["chunk_size"]]).then(fn = update_slider, inputs = [chunk_summary_state], outputs = [upload_components["chunk_sum"]])
        log_tab.select()


    return Technomancer


if __name__ in "__main__":
    Technomancer = launch_technomancer()
    Technomancer.launch(server_name = server_name, server_port = server_port)