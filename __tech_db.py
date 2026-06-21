from os.path import basename
from __log_fn import setup_logs
import logging

import gradio as gr

from __rag_pipeline import (create_collection, 
                            delete_document, delete_collection, 
                            find_collections, find_documents,
                            get_metadata,
                            load_documents,
                            update_metadata)

from __states import (chunk_overlap_state, chunk_size_state, chunk_summary_state, chunk_batches_state,
                      documents_list_state, 
                      embed_model_state, 
                      k_state,
                      lang_model_state, 
                      named_chunkoverlap_state, name_chunksize_state, name_embed_state, name_rule_state, name_lang_state, name_chunksum_state, name_chunkbatch_state,
                      percent_state, 
                      rule_system_state, rule_systems_list_state, 
                      tags_list_state, true_state,
                      save_chunk_state, save_sum_state,
                      upload_status_state)

from __tech_fn import (append_state_list, 
                       change_state, change_state_list, chunking_type,
                       export_tags, 
                       list_length, 
                       update_slider, update_textbox,  update_drop_down)


def create_db():
    '''
    '''
    
    with gr.Blocks(fill_height = True) as db:

        # local_embedding_state = gr.State(value = "")  # this is just because I don't want to import gradio to __rag_pipeline.py
        
        # with gr.Row():

        with gr.Accordion(label = "Database of Holding Management", open = True) as DBoH_acc:

            with gr.Row(variant="panel"):
                available_rule_systems_dd = gr.Dropdown(choices = [], label = "Choose Rule System", interactive = True, scale = 10)
                refresh_rules_btn_2 = gr.Button(value = "Refresh Rules List", scale = 1)
                available_documents_textbox = gr.Textbox(label = "Docs Found", value = "", scale = 1)
            
            with gr.Row(variant="panel"):
                available_documents_dd = gr.Dropdown(choices = [], label = "List of available documents in selected collection", interactive = True, scale = 15)
                delete_document_btn = gr.Button(value = "Delete Selected Document", scale = 1)

            with gr.Row(variant="panel"):
                more_metadata_dd = gr.Dropdown(label = "All metadata tags on current document.", choices = [], interactive = False, multiselect = True, scale = 15)
                with gr.Column():
                    # more_metadata_btn = gr.Button(value = "Add Metadata", scale = 1)
                    enrich_document_btn = gr.Button(value = "Auto Apply Metadata", scale = 1)
                    gen_summary_btn = gr.Button(value = "Regenerate Summary", scale = 1)
                local_embedding_box = gr.Textbox(label = "Embedding used for document", value = "", scale = 3, interactive = False)

            with gr.Row(variant="panel"):
                chunks_dd = gr.Dropdown(label = "Document Chunk IDs")
            with gr.Row(variant="panel"):
                chunk_data_box = gr.TextArea(value = "Document Data", info = "Can be edited", scale = 10)
                chunk_data_btn = gr.Button(value = "Update Text", scale = 1)
            with gr.Row(variant="panel"):
                with gr.Column():
                    chunk_tags_dd = gr.Dropdown(choices = [], label = "Metadata Tags", interactive = True, multiselect = True, allow_custom_value = True)
                    chunk_tags_add_btn = gr.Button(value = "Add Metadata Tags")
                with gr.Column():
                    chunks_metadata_dd = gr.Dropdown(choices = [], label = "Metadatas")
                    chunks_metadata_box = gr.Textbox(label = "Metadatas value", interactive = True)

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

                
        with gr.Accordion(label = "Quarentined Documents", open = False) as quarentine:
            gr.Markdown("This is for quarentined documents with low QS.")
        
            # with gr.Row():
            #     with gr.Column():
            #         gr.Markdown("Embedding Model choice. If you have multiple databases, this can be used with different embedding models and switch between the two. Not fully implemented yet.")
            #         embed_models_dd = gr.Dropdown(label = "Embedding Models", choices = [], info = "Choice of embedding model. Only for advanced uses.", interactive = True)
                
            #     with gr.Column():
            #         gr.Markdown("Langauge choice for summary of sections. Consider keeping it with your primary language model, but if you want to use a different one, go ahead.")
            #         lang_model_sum_dd = gr.Dropdown(label = "Language Models", choices = [], interactive = True)
                
        # add_doc_tags.click(fn = change_state_list, inputs = [metadata_tags_dd], outputs = [document_tags_s])
        # add_doc_tags.click(fn = change_state_list, inputs = [metadata_tags_dd], outputs = [tags_list_state])

        # available_rule_systems_dd.select(fn = find_documents, inputs = [available_rule_systems_dd], outputs = [documents_list_state]).then(fn = update_drop_down, inputs = [documents_list_state], outputs = [available_documents_dd]).then(fn = list_length, inputs = [documents_list_state], outputs = [available_documents_textbox])
        # available_documents_dd.select(fn = get_metadata, inputs = [available_rule_systems_dd, available_documents_dd], outputs = [more_metadata_dd, local_embedding_state]).then(fn = update_textbox, inputs = [local_embedding_state], outputs = [local_embedding_box])
        
        # c_size_slide.change(fn = change_state, inputs = [c_size_slide, chunk_size_state, true_state, name_chunksize_state], outputs = [chunk_size_state]).then(fn = update_slider, inputs = [chunk_size_state, percent_state], outputs = [c_overlap_slide]).then(fn = change_state, inputs = [c_overlap_slide, chunk_overlap_state, true_state, named_chunkoverlap_state], outputs = [chunk_overlap_state])  # This might not work. I'm not confident it will.
        # c_overlap_slide.change(fn = change_state, inputs = [c_overlap_slide, chunk_overlap_state, true_state, named_chunkoverlap_state], outputs = [chunk_overlap_state])  # this used to update both the chunk size and overlap. Now this just updates the overlap.
        # c_summary_slide.change(fn = change_state, inputs = [c_summary_slide, chunk_summary_state, true_state, name_chunksum_state], outputs = [chunk_summary_state])
        # c_batch_slide.change(fn = change_state, inputs = [c_batch_slide, chunk_batches_state, true_state, name_chunkbatch_state], outputs = [chunk_batches_state])
        # chunking_type_check.input(fn = chunking_type, inputs = [chunking_type_check], outputs = [save_chunk_state, save_sum_state, chunking_type_check])

        del_collection_btn.click(fn = delete_collection, inputs = [available_rule_systems_dd], outputs = [rule_system_state]).then(fn = find_collections, outputs = [rule_systems_list_state]).then(fn = update_drop_down, inputs = [rule_systems_list_state], outputs = [available_rule_systems_dd])
        # delete_document_btn.click(fn = delete_document, inputs = [available_rule_systems_dd, available_documents_dd]).then(fn = find_documents, inputs = [available_rule_systems_dd], outputs = [documents_list_state]).then(fn = update_drop_down, inputs = [documents_list_state], outputs = [available_documents_dd])
        # # del_everything_btn.click(fn = delete_collection, inputs = [available_rule_systems_dd, true_state]).then(fn = find_collections, outputs = [rule_system_state]).then(fn = update_drop_down, inputs = [rule_system_state], outputs = [rule_systems_dd]).then(fn = update_drop_down, inputs = [rule_system_state], outputs = [available_documents_dd])

        # embed_models_dd.select(fn = change_state, inputs = [embed_models_dd, embed_model_state, true_state, name_embed_state], outputs = [embed_model_state]).then(fn = update_textbox, inputs = [embed_model_state], outputs = [embed_textbox])

        # more_metadata_btn.click(fn = update_metadata, inputs = [available_rule_systems_dd, available_documents_dd, more_metadata_dd])

        # rule_system_add_btn.click(fn = change_state, inputs = [rule_systems_dd, rule_system_state, true_state, name_rule_state], outputs = [rule_system_state]).then(fn = create_collection, inputs = [rule_system_state]).then(fn = append_state_list, inputs = [rule_systems_list_state, rule_system_state], outputs = [rule_systems_list_state]).then(fn = update_drop_down, inputs = [rule_systems_list_state, rule_system_state], outputs = [rule_systems_dd]).then(fn = update_drop_down, inputs = [rule_systems_list_state, rule_system_state], outputs = [available_rule_systems_dd]).then(update_textbox, inputs = [rule_system_state], outputs = [selected_rule_system])

        # refresh_rules_btn_2.click(fn = find_collections, outputs = [rule_systems_list_state]).then(fn = update_drop_down, inputs = [rule_systems_list_state], outputs = [available_rule_systems_dd]).then(fn = update_drop_down, inputs = [rule_systems_list_state], outputs = [rule_systems_dd])
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
        # "lang_model_sum_dd": lang_model_sum_dd,
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

    TECH_MANAGEMENT, mangagement_components = create_db(["DB Path"], ["Embedding Choice"], ["AD&D", "Shadowrun", "Rifts"], ["Tag 1", "Tag 2", "Tag 3"])
else:
    logger = logging.getLogger(__name__)
    print("Rendered Management Tab")
    logger.info("Rendered Management Tab @ (time to be implemented)")
