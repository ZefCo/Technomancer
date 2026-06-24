from os.path import basename
from __log_fn import setup_logs
import logging

import gradio as gr

from __rag_pipeline import (
                            create_collection, 
                            delete_document, delete_collection, 
                            find_chunk, find_chunks, find_collections, find_documents,
                            generate_summary, get_metadata, 
                            load_documents,
                            update_metadata
                            )

from __states import (
                      embed_models_list_state,
                      lang_models_list_state,
                      rule_system_state, rule_systems_list_state, 
                      true_state
                      )

from __tech_fn import (
                       append_state_list, 
                       change_state, change_state_list, chunking_type,
                       export_tags, 
                       list_length, 
                       update_textarea, update_textbox,  update_drop_down
                       )


def create_db():
    '''
    '''
    documents_list_state = gr.State(value = [])
    chunk_list_state = gr.State(value = [])
    empty_text_state = gr.State(value = "")  # I might be able to remove this too and just use the None state.
    none_state = gr.State(value = None)  # in case I need to pass none
    # metadata_tags_list_state = gr.State(value = [])

    with gr.Blocks(fill_height = True) as db:

        # with gr.Accordion(label = "Database of Holding Management", open = True) as DBoH_acc:
        gr.Markdown("/* Not yet implemented")

        with gr.Row(variant="panel"):
            available_rule_systems_dd = gr.Dropdown(choices = [], label = "Choose Rule System", interactive = True, scale = 10)
            refresh_rules_btn = gr.Button(value = "Refresh Rules List", scale = 1)
            available_documents_textbox = gr.Textbox(label = "Docs Found", value = "", scale = 1)
        
        with gr.Row(variant="panel"):
            available_documents_dd = gr.Dropdown(choices = [], label = "List of available documents in selected rule system", interactive = True, scale = 15)
            delete_document_btn = gr.Button(value = "Delete Selected Document", scale = 1)

        with gr.Row(variant="panel"):
            all_metadata_dd = gr.Dropdown(label = "All metadata tags on current document.", choices = [], interactive = False, multiselect = True, scale = 15, allow_custom_value = True)
            enrich_document_btn = gr.Button(value = "Auto Apply Metadata*", scale = 1)
        with gr.Row(variant= "panel"):
            local_embedding_box = gr.Textbox(label = "Embedding used for document", value = "", interactive = False)
            local_lang_dd = gr.Dropdown(label = "Language Model for regenerating chunks", choices = [], interactive = True)
            local_vis_dd = gr.Dropdown(label = "Vision Model for regenerating chunks", choices = [], interactive = True)
            local_chunk_summary_num = gr.Number(label = "Chunks to summarize", value = 10, precision = 0)
            gen_summary_btn = gr.Button(value = "Regenerate Summary")
            generate_status_box = gr.Textbox(label = "Status*", value = "Warning, there are no status updates as of yet")

        with gr.Row(variant="panel"):
            chunks_dd = gr.Dropdown(label = "Document Chunk IDs", interactive = True, scale = 10)
            chunks_len_box = gr.Textbox(label = "Number of Chunks", scale = 1)
        with gr.Row(variant="panel"):
            chunk_data_area = gr.TextArea(label = "Document Data", info = "Can be edited*", scale = 10)
            with gr.Column():
                chunk_update_btn = gr.Button(value = "Update Text*", scale = 1)
                chunk_del_btn = gr.Button(value = "Delete Chunk*", scale = 1)
        with gr.Row(variant="panel"):
            with gr.Column():
                with gr.Row():
                    page_location_box = gr.Textbox(label = "Page Source", interactive = False, value = "")
                    chunk_type_box = gr.Textbox(label = "Chunk Type", interactive = False, value = "")
                    extraction_method_box = gr.Textbox(label = "Extraction Method:", interactive = False, value = "")
                with gr.Row():
                    quality_score_box = gr.Textbox(label = "Chunk Quality Score", interactive = False, value = "")
                    angle_score_box = gr.Textbox(label = "Angeled Score (1.0 is perfectly vertical)", interactive = False, value = "")
                    double_score_box = gr.Textbox(label = "Doubled Letter Ratio", interactive = False, value = "")
                    ave_word_score_box = gr.Textbox(label = "Average Words", interactive = False, value = "")
                    word_len_score_box = gr.Textbox(label = "Suspicious Word Length", interactive = False, value = "")
                    text_len_score_box = gr.Textbox(label = "Text Length", interactive = False, value = "")
                    has_images_box = gr.Textbox(label = "Has Images", interactive = False, value = "")


        with gr.Row(variant="panel"):
            with gr.Column():
                chunk_tags_dd = gr.Dropdown(choices = [], label = "Metadata Tags", interactive = True, multiselect = True, allow_custom_value = True)
                chunk_tags_add_btn = gr.Button(value = "Add Metadata Tags*")

        with gr.Row(variant="panel"):
            with gr.Accordion(label = "Delete Entire Rule System", open = False):
                gr.Markdown("This will delete the entire rule system, along with all books and documents associated with it. Select the rule system at the top and press the big button here. It *cannot* be undone. All documents, chunks, anything stored under that rule system will be removed.\nTo delete the entire database: you must manually delete it from your system. Because no one wants to accidentally delete their database, that has to be done manually.")
                del_collection_btn = gr.Button(value = "Removes Rule System (The Nuclear Option)")
                
        # with gr.Accordion(label = "Quarantined Documents", open = False) as quarantine:
        #     gr.Markdown("This is for quarantined documents with low QS.")

        available_rule_systems_dd.select(
                   fn = find_documents, inputs = [available_rule_systems_dd], outputs = [documents_list_state]
            ).then(fn = update_drop_down, inputs = [none_state], outputs = [available_documents_dd]
            ).then(fn = update_drop_down, inputs = [documents_list_state], outputs = [available_documents_dd]
            ).then(fn = list_length, inputs = [documents_list_state], outputs = [available_documents_textbox]
            ).then(fn = update_drop_down, inputs = [none_state], outputs = [all_metadata_dd]
            ).then(fn = update_drop_down, inputs = [none_state], outputs = [chunks_dd]
            ).then(fn = update_textarea, inputs = [empty_text_state], outputs = [chunk_data_area]
            ).then(fn = update_textbox, inputs = [empty_text_state], outputs = [chunks_len_box]
            ).then(fn = update_textbox, inputs = [empty_text_state], outputs = [page_location_box]
            ).then(fn = update_textbox, inputs = [empty_text_state], outputs = [chunk_type_box]
            ).then(fn = update_textbox, inputs = [empty_text_state], outputs = [extraction_method_box]
            ).then(fn = update_textbox, inputs = [empty_text_state], outputs = [quality_score_box]
            ).then(fn = update_textbox, inputs = [empty_text_state], outputs = [angle_score_box]
            ).then(fn = update_textbox, inputs = [empty_text_state], outputs = [double_score_box]
            ).then(fn = update_textbox, inputs = [empty_text_state], outputs = [ave_word_score_box]
            ).then(fn = update_textbox, inputs = [empty_text_state], outputs = [word_len_score_box]
            ).then(fn = update_textbox, inputs = [empty_text_state], outputs = [text_len_score_box]
            ).then(fn = update_textbox, inputs = [empty_text_state], outputs = [has_images_box]
            )
        
        available_documents_dd.select(
                   fn = get_metadata, inputs = [available_rule_systems_dd, available_documents_dd], outputs = [all_metadata_dd, local_embedding_box]
            ).then(fn = find_chunks, inputs = [available_rule_systems_dd, available_documents_dd], outputs = [chunk_list_state, chunks_len_box]
            ).then(fn = update_drop_down, inputs = [chunk_list_state], outputs = [chunks_dd]
            ).then(fn = update_textarea, inputs = [empty_text_state], outputs = [chunk_data_area]
            ).then(fn = update_textbox, inputs = [empty_text_state], outputs = [page_location_box]
            ).then(fn = update_textbox, inputs = [empty_text_state], outputs = [chunk_type_box]
            ).then(fn = update_textbox, inputs = [empty_text_state], outputs = [extraction_method_box]
            ).then(fn = update_textbox, inputs = [empty_text_state], outputs = [quality_score_box]
            ).then(fn = update_textbox, inputs = [empty_text_state], outputs = [angle_score_box]
            ).then(fn = update_textbox, inputs = [empty_text_state], outputs = [double_score_box]
            ).then(fn = update_textbox, inputs = [empty_text_state], outputs = [ave_word_score_box]
            ).then(fn = update_textbox, inputs = [empty_text_state], outputs = [word_len_score_box]
            ).then(fn = update_textbox, inputs = [empty_text_state], outputs = [text_len_score_box]
            ).then(fn = update_textbox, inputs = [empty_text_state], outputs = [has_images_box]
            )
        
        chunks_dd.select(
                   fn = find_chunk, inputs = [available_rule_systems_dd, chunks_dd], outputs = [chunk_data_area, chunk_tags_dd, page_location_box, quality_score_box, chunk_type_box, extraction_method_box, angle_score_box, double_score_box, ave_word_score_box, word_len_score_box, text_len_score_box, has_images_box]
            )

        del_collection_btn.click(
                   fn = delete_collection, inputs = [available_rule_systems_dd], outputs = [rule_system_state]
            ).then(fn = find_collections, outputs = [rule_systems_list_state]
            ).then(fn = update_drop_down, inputs = [rule_systems_list_state], outputs = [available_rule_systems_dd]
            ).then(fn = list_length, inputs = [empty_text_state], outputs = [available_documents_textbox]
            ).then(fn = update_drop_down, inputs = [none_state], outputs = [all_metadata_dd]
            ).then(fn = update_drop_down, inputs = [none_state], outputs = [chunks_dd]
            ).then(fn = update_textbox, inputs = [empty_text_state], outputs = [local_embedding_box]
            ).then(fn = update_textarea, inputs = [empty_text_state], outputs = [chunk_data_area]
            ).then(fn = update_textbox, inputs = [empty_text_state], outputs = [chunks_len_box]
            ).then(fn = update_textbox, inputs = [empty_text_state], outputs = [page_location_box]
            ).then(fn = update_textbox, inputs = [empty_text_state], outputs = [chunk_type_box]
            ).then(fn = update_textbox, inputs = [empty_text_state], outputs = [extraction_method_box]
            ).then(fn = update_textbox, inputs = [empty_text_state], outputs = [quality_score_box]
            ).then(fn = update_textbox, inputs = [empty_text_state], outputs = [angle_score_box]
            ).then(fn = update_textbox, inputs = [empty_text_state], outputs = [double_score_box]
            ).then(fn = update_textbox, inputs = [empty_text_state], outputs = [ave_word_score_box]
            ).then(fn = update_textbox, inputs = [empty_text_state], outputs = [word_len_score_box]
            ).then(fn = update_textbox, inputs = [empty_text_state], outputs = [text_len_score_box]
            ).then(fn = update_textbox, inputs = [empty_text_state], outputs = [has_images_box]
            )
        
        delete_document_btn.click(
                   fn = delete_document, inputs = [available_rule_systems_dd, available_documents_dd]
            ).then(fn = find_documents, inputs = [available_rule_systems_dd], outputs = [documents_list_state]
            ).then(fn = update_drop_down, inputs = [documents_list_state], outputs = [available_documents_dd]
            ).then(fn = list_length, inputs = [documents_list_state], outputs = [available_documents_textbox]
            ).then(fn = update_drop_down, inputs = [none_state], outputs = [all_metadata_dd]
            ).then(fn = update_drop_down, inputs = [none_state], outputs = [chunks_dd]
            ).then(fn = update_textbox, inputs = [empty_text_state], outputs = [local_embedding_box]
            ).then(fn = update_textarea, inputs = [empty_text_state], outputs = [chunk_data_area]
            ).then(fn = update_textbox, inputs = [empty_text_state], outputs = [chunks_len_box]
            ).then(fn = update_textbox, inputs = [empty_text_state], outputs = [page_location_box]
            ).then(fn = update_textbox, inputs = [empty_text_state], outputs = [chunk_type_box]
            ).then(fn = update_textbox, inputs = [empty_text_state], outputs = [extraction_method_box]
            ).then(fn = update_textbox, inputs = [empty_text_state], outputs = [quality_score_box]
            ).then(fn = update_textbox, inputs = [empty_text_state], outputs = [angle_score_box]
            ).then(fn = update_textbox, inputs = [empty_text_state], outputs = [double_score_box]
            ).then(fn = update_textbox, inputs = [empty_text_state], outputs = [ave_word_score_box]
            ).then(fn = update_textbox, inputs = [empty_text_state], outputs = [word_len_score_box]
            ).then(fn = update_textbox, inputs = [empty_text_state], outputs = [text_len_score_box]
            ).then(fn = update_textbox, inputs = [empty_text_state], outputs = [has_images_box]
            )

        gen_summary_btn.click(
                   fn = generate_summary, inputs = [available_rule_systems_dd, available_documents_dd, local_embedding_box, local_lang_dd, local_chunk_summary_num]
            )

        refresh_rules_btn.click(
                   fn = find_collections, outputs = [rule_systems_list_state]
            ).then(fn = update_drop_down, inputs = [rule_systems_list_state], outputs = [available_rule_systems_dd]
            )

    return db, {
        "lang_model_sum_dd": local_lang_dd,
        "rule_systems_dd": available_rule_systems_dd, 
        "vis_model_dd": local_vis_dd
        }


if __name__ in "__main__":
    from datetime import datetime
    import pathlib

    cwd = pathlib.Path.cwd()    
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")    

    log_dir = cwd / "Logs"
    log_dir.mkdir(parents = True, exist_ok = True)
    
    logger = setup_logs(__name__, log_dir / f"{pathlib.Path(basename(__file__)).stem}_{timestamp}.log")

    TECH_MANAGEMENT, management_components = create_db(["DB Path"], ["Embedding Choice"], ["AD&D", "Shadowrun", "Rifts"], ["Tag 1", "Tag 2", "Tag 3"])
else:
    logger = logging.getLogger(__name__)
    print("Rendered Management Tab")
    logger.info("Rendered Management Tab @ (time to be implemented)")
