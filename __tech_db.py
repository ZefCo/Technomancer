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
                       update_slider, update_textbox,  update_drop_down
                       )


def create_db():
    '''
    '''
    documents_list_state = gr.State(value = [])
    chunk_list_state = gr.State(value = [])
    # metadata_tags_list_state = gr.State(value = [])
    with gr.Blocks(fill_height = True) as db:

        # local_embedding_state = gr.State(value = "")  # this is just because I don't want to import gradio to __rag_pipeline.py
        
        # with gr.Row():

        with gr.Accordion(label = "Database of Holding Management", open = True) as DBoH_acc:
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
                local_chunk_summary_num = gr.Number(label = "Chunks to summarize", value = 10, precision = 0)
                gen_summary_btn = gr.Button(value = "Regenerate Summary")
                generate_status_box = gr.Textbox(label = "Status*", value = "Warning, there are no status updates as of yet")
                # with gr.Column():
                    # more_metadata_btn = gr.Button(value = "Add Metadata", scale = 1)
                # with gr.Column():

            with gr.Row(variant="panel"):
                chunks_dd = gr.Dropdown(label = "Document Chunk IDs", interactive = True, scale = 10)
                chunks_len_box = gr.Textbox(label = "Number of Chunks", scale = 1)
            with gr.Row(variant="panel"):
                chunk_data_area = gr.TextArea(label = "Document Data", info = "Can be edited*", scale = 10)
                with gr.Column():
                    chunk_data_btn = gr.Button(value = "Update Text*", scale = 1)
                    chunk_del_btn = gr.Button(value = "Delete Chunk*", scale = 1)
                    quality_score_box = gr.Textbox(label = "Quality Score", interactive = False, value = "")
                    page_location_box = gr.Textbox(label = "Page Source", interactive = False, value = "")
                    chunk_type_box = gr.Textbox(label = "Metadatas*", interactive = False, value = "")

            with gr.Row(variant="panel"):
                with gr.Column():
                    chunk_tags_dd = gr.Dropdown(choices = [], label = "Metadata Tags*", interactive = True, multiselect = True, allow_custom_value = True)
                    chunk_tags_add_btn = gr.Button(value = "Add Metadata Tags*")

            # with gr.Row(variant="panel"):
            #     gr.Markdown("Note on Deletion: This deletes all things based on the file name of the book. If the book name is close enough to [an]other book[s], it is possible that the other book[s] will be delete too! Check to make sure that the books you want to stay in the database are in fact still there!")

            with gr.Row(variant="panel"):
                with gr.Accordion(label = "Delete Entire Rule System", open = False):
                    gr.Markdown("This will delete the entire rule system, along with all books and documents associated with it. It *cannot* be undone.\nYou can Remove a given rule set, clearing out the entire set of documents!\nTo delete the entire database: you must manually delete it from your system. Yes it is possible to add that functionality here, but because no one wants to accidentally delete their database, that has to be done manually.")
                    del_collection_btn = gr.Button(value = "Removes Rule System (The Nuclear Option)")
                    # clear_collection_btn = gr.Button(value = "Clears a Rule System of all documents (empties rule system - not yet implemented)")
                    # del_everything_btn = gr.Button(value = "Reset entire database (the BIGGER red button)")

        with gr.Accordion(label = "Document Chunks", open = False) as doc_chunks:
            gr.Markdown("This is for listing all the chunks of the selected document")

                
        with gr.Accordion(label = "Quarantined Documents", open = False) as quarantine:
            gr.Markdown("This is for quarantined documents with low QS.")
        
                
        # add_doc_tags.click(fn = change_state_list, inputs = [metadata_tags_dd], outputs = [document_tags_s])
        # add_doc_tags.click(fn = change_state_list, inputs = [metadata_tags_dd], outputs = [tags_list_state])

        available_rule_systems_dd.select(fn = find_documents, inputs = [available_rule_systems_dd], outputs = [documents_list_state]).then(fn = update_drop_down, inputs = [documents_list_state], outputs = [available_documents_dd]).then(fn = list_length, inputs = [documents_list_state], outputs = [available_documents_textbox])
        available_documents_dd.select(fn = get_metadata, inputs = [available_rule_systems_dd, available_documents_dd], outputs = [all_metadata_dd, local_embedding_box]).then(fn = find_chunks, inputs = [available_rule_systems_dd, available_documents_dd], outputs = [chunk_list_state, chunks_len_box]).then(fn = update_drop_down, inputs = [chunk_list_state], outputs = [chunks_dd])
        
        chunks_dd.select(fn = find_chunk, inputs = [available_rule_systems_dd, chunks_dd], outputs = [chunk_data_area, chunk_tags_dd, page_location_box, chunk_type_box, quality_score_box])

        del_collection_btn.click(fn = delete_collection, inputs = [available_rule_systems_dd], outputs = [rule_system_state]).then(fn = find_collections, outputs = [rule_systems_list_state]).then(fn = update_drop_down, inputs = [rule_systems_list_state], outputs = [available_rule_systems_dd])
        delete_document_btn.click(fn = delete_document, inputs = [available_rule_systems_dd, available_documents_dd]).then(fn = find_documents, inputs = [available_rule_systems_dd], outputs = [documents_list_state]).then(fn = update_drop_down, inputs = [documents_list_state], outputs = [available_documents_dd]).then(fn = list_length, inputs = [documents_list_state], outputs = [available_documents_textbox])
        # # del_everything_btn.click(fn = delete_collection, inputs = [available_rule_systems_dd, true_state]).then(fn = find_collections, outputs = [rule_system_state]).then(fn = update_drop_down, inputs = [rule_system_state], outputs = [rule_systems_dd]).then(fn = update_drop_down, inputs = [rule_system_state], outputs = [available_documents_dd])

        # embed_models_dd.select(fn = change_state, inputs = [embed_models_dd, embed_model_state, true_state, name_embed_state], outputs = [embed_model_state]).then(fn = update_textbox, inputs = [embed_model_state], outputs = [embed_textbox])

        gen_summary_btn.click(fn = generate_summary, inputs = [available_rule_systems_dd, available_documents_dd, local_embedding_box, local_lang_dd, local_chunk_summary_num])

        # more_metadata_btn.click(fn = update_metadata, inputs = [available_rule_systems_dd, available_documents_dd, all_metadata_dd])

        # rule_system_add_btn.click(fn = change_state, inputs = [rule_systems_dd, rule_system_state, true_state, name_rule_state], outputs = [rule_system_state]).then(fn = create_collection, inputs = [rule_system_state]).then(fn = append_state_list, inputs = [rule_systems_list_state, rule_system_state], outputs = [rule_systems_list_state]).then(fn = update_drop_down, inputs = [rule_systems_list_state, rule_system_state], outputs = [rule_systems_dd]).then(fn = update_drop_down, inputs = [rule_systems_list_state, rule_system_state], outputs = [available_rule_systems_dd]).then(update_textbox, inputs = [rule_system_state], outputs = [selected_rule_system])

        refresh_rules_btn.click(fn = find_collections, outputs = [rule_systems_list_state]).then(fn = update_drop_down, inputs = [rule_systems_list_state], outputs = [available_rule_systems_dd])
        # refresh_rules_btn_1.click(fn = find_collections, outputs = [rule_systems_list_state]).then(fn = update_drop_down, inputs = [rule_systems_list_state], outputs = [available_rule_systems_dd]).then(fn = update_drop_down, inputs = [rule_systems_list_state], outputs = [rule_systems_dd])
        # rule_systems_dd.select(fn = change_state, inputs = [rule_systems_dd, rule_system_state, true_state, name_rule_state], outputs = [rule_system_state]).then(fn = update_textbox, inputs = [rule_system_state], outputs = [selected_rule_system])
        
        # save_tags.click(fn = append_state_list, inputs = [tags_list_state, metadata_tags_dd], outputs = [tags_list_state]).then(fn = update_drop_down, inputs = [tags_list_state], outputs = [metadata_tags_dd]).then(fn = export_tags, inputs = [tags_list_state])

        # lang_model_sum_dd.select(fn = change_state, inputs = [lang_model_sum_dd, lang_model_state, true_state, name_lang_state], outputs = [lang_model_state])

        # upload_file_space.upload(fn = change_state, inputs = [rule_systems_dd], outputs = [rule_system_state]).then(fn = append_state_list, inputs = [rule_systems_list_state, rule_system_state], outputs = [rule_systems_list_state]).then(fn = update_drop_down, inputs = [rule_systems_list_state, rule_system_state], outputs = [rule_systems_dd]).then(fn = update_drop_down, inputs = [rule_systems_list_state, rule_system_state], outputs = available_rule_systems_dd).then(fn = load_documents, inputs = [upload_file_space, rule_system_state, embed_model_state, lang_model_state, metadata_tags_dd, chunk_size_state, chunk_overlap_state, chunk_batches_state, chunk_summary_state, save_chunk_state, save_sum_state], outputs = [upload_status_box])

    return db, {
        # "chunk_batch": c_batch_slide,
        # "chunk_overlap": c_overlap_slide,
        # "chunk_size": c_size_slide,
        # "chunk_sum": c_summary_slide,
        # "embed_models_dd": embed_models_dd, 
        # "embed_textbox": embed_textbox, 
        "lang_model_sum_dd": local_lang_dd,
        # "metadata_tags_dd": metadata_tags_dd, 
        "rule_systems_dd": available_rule_systems_dd, 
        # "rule_systems_dd_2": rule_systems_dd, 
        # "upload_status_box": upload_status_box
        }
    # return upload


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
