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
                      settings_path_tags_state,
                      upload_status_state)

from __tech_fn import (append_state_list, 
                       change_state, change_state_list, 
                       export_tags, 
                       list_length, 
                       update_slider, update_textbox,  update_drop_down)


def create_upload():
    '''
    '''
    
    with gr.Blocks() as upload:
        
        with gr.Row(equal_height = True):
                        
            with gr.Column(variant = "panel"):
            
                # with gr.Row():
                gr.Markdown("## Rule System & Tags")
                gr.Markdown("All documents need to be added to a rule system, a group that the document can belong to. To create a rule system group, select from or type into the dropdown menu. Tags can be added to any document, which will make searching those documents with Technomancer easier. The Save Tags button exports the tags to an external file, which loads them later. To delete tags, they are saved in the Settings folder in the Tags.yaml file: simple remove the lines of tags you don't want nor need. Alternativly, add to that file as you wish.")
                with gr.Column(variant = "panel"):
                    with gr.Row():
                        rule_systems_dd = gr.Dropdown(label = "Rule System to Upload to (Required)", choices = [], interactive = True, allow_custom_value = True, scale = 3)
                        rule_system_add_btn = gr.Button(value = "Add Rule System", scale = 1)

                    # new_rule_system = gr.Textbox(label = None, submit_btn = True, placeholder = "Type in a new rule system/collection")
                
                with gr.Row():
                    metadata_tags_dd = gr.Dropdown(choices = [], label = "Tags to organize documents", multiselect = True, allow_custom_value = True, scale = 5, interactive = True)
                    save_tags = gr.Button(value = "Save Tags", scale = 1)

            with gr.Column(variant = "panel"):
                gr.Markdown("## Chunks")

                gr.Markdown("While this can go as high as 1,000 chunks, consider staying below 512, with overlap being about 0.10 - 0.20 of the chunk size. Chunk batches is for how many chunks will be added at a time. Due to the size of the books, there can be several thousand chunks that are needed to be loaded. The Summary Chunks is how many chunks are bundeled together to create a larger summary of information. Useful for getting large concepts of rules.")
                c_size_slide = gr.Slider(minimum = 10, maximum = 1_000, value = 10, label = "Chunk Size", interactive = True, precision = 0)
                c_overlap_slide = gr.Slider(minimum = 1, maximum = 500, value = 10, label = "Chunk Overlap", interactive = True, precision = 0)
                c_batch_slide = gr.Slider(minimum = 1, maximum = 200, value = 10, label = "Chunk Batches", interactive = True, precision = 0)
                c_summary_slide = gr.Slider(minimum = 1, maximum = 100, value = 10, label = "Summary Chunks", interactive = True, precision = 0)
                # embed_textbox = gr.Textbox(label = "Embedding Model Being Used", value = "")

            with gr.Column(variant = "panel"):
                gr.Markdown("## File Upload")

                gr.Markdown("Supported file types: .pdf, .docx, .txt, .csv")
                with gr.Column(variant = "compact"):
                    with gr.Row():
                        selected_rule_system = gr.Textbox(label = "Add to rule system (adjusted in Rule System & Tags column)", interactive = False, value = "Select Rule system")
                        embed_textbox = gr.Textbox(label = "w/ Embedding Model: (adjusted in Advanced Settings)", interactive = False, value = "")
                upload_file_space = gr.File(label = f"Drag and drop a file")
                upload_status_box = gr.Textbox(label  = "Upload Status", interactive = False, value = "No file uploaded yet")

        with gr.Row():

            with gr.Accordion(label = "Database of Holding Management", open = False) as DBoH_acc:

                with gr.Accordion(label = "Manage Documents", open = False) as man_DBoH_acc:

                    with gr.Row():
                        gr.Markdown("Note on Deletion: This deletes all things based on the file name of the book. If the book name is close enough to [an]other book[s], it is possible that the other book[s] will be delete too! Check to make sure that the books you want to stay in the database are in fact still there!")

                    with gr.Row():
                            available_rule_systems_dd = gr.Dropdown(choices = [], label = "Choose Collection", interactive = True, scale = 10)
                            available_documents_textbox = gr.Textbox(label = "Docs Found", value = "", scale = 1)

                            available_documents_dd = gr.Dropdown(choices = [], label = "List of available documents in selected collection", interactive = True, scale = 15)

                            delete_document_btn = gr.Button(value = "Delete Selected Document", scale = 1)

                    with gr.Row():
                        more_metadata_dd = gr.Dropdown(label = "Tags on current document, select additional tags. This does not update the master list, only the document selected.", choices = [], interactive = True, multiselect = True, scale = 15, allow_custom_value = True)
                        more_metadata_btn = gr.Button(value = "Add Metadata")

                    with gr.Accordion(label = "Delete Entire Rule System", open = False):
                        gr.Markdown("This will delete the entire rule system, along with all books and documents associated with it. It *cannot* be undone.\nYou can Remove a given rule set, clearing out the entire set of documents!\nTo delete the entire database: you must manually delete it from your system. Yes it is possible to add that functionality here, but because no one wants to accidentally delete their database, that has to be done manually.")
                        del_collection_btn = gr.Button(value = "Removes Rule System (The Nuclear Option)")
                        # clear_collection_btn = gr.Button(value = "Clears a Rule System of all documents (empties rule system - not yet implemented)")
                        # del_everything_btn = gr.Button(value = "Reset entire database (the BIGGER red button)")
                
                with gr.Accordion(label = "Advanced Settings", open = False) as adv_DBoH_acc:
                
                    with gr.Row():
                        with gr.Column():
                            gr.Markdown("Embedding Model choice. If you have multiple databases, this can be used with different embedding models and switch between the two. Not fully implemented yet.")
                            embed_models_dd = gr.Dropdown(label = "Embedding Models", choices = [], info = "Choice of embedding model. Only for advanced uses.", interactive = True)

                        with gr.Column():
                            gr.Markdown("Note this has not be implemented yet as I need to figure out how to pass all this information to and from Gradio. Will use Settings files probably.")
                            list_of_db_dd = gr.Dropdown(info =  "Not implemented yet", label = "Choose Database", choices = [], interactive = True)
                            db_path_space = gr.Markdown("Reserved for Create Path to Database option. (Not implemented yet. Button? Textbox? Dropdowns? Not sure what to do here.)")
                        
                        with gr.Column():
                            gr.Markdown("Langauge choice for summary of sections. Consider keeping it with your primary language model, but if you want to use a different one, go ahead.")
                            lang_model_sum_dd = gr.Dropdown(label = "Language Models", choices = [], interactive = True)
                
        # add_doc_tags.click(fn = change_state_list, inputs = [metadata_tags_dd], outputs = [document_tags_s])
        # add_doc_tags.click(fn = change_state_list, inputs = [metadata_tags_dd], outputs = [tags_list_state])

        available_rule_systems_dd.select(fn = find_documents, inputs = [available_rule_systems_dd], outputs = [documents_list_state]).then(fn = update_drop_down, inputs = [documents_list_state], outputs = [available_documents_dd]).then(fn = list_length, inputs = [documents_list_state], outputs = [available_documents_textbox])
        available_documents_dd.select(fn = get_metadata, inputs = [available_rule_systems_dd, available_documents_dd], outputs = [more_metadata_dd])
        
        c_size_slide.change(fn = change_state, inputs = [c_size_slide, chunk_size_state, true_state, name_chunksize_state], outputs = [chunk_size_state]).then(fn = update_slider, inputs = [chunk_size_state, percent_state], outputs = [c_overlap_slide]).then(fn = change_state, inputs = [c_overlap_slide, chunk_overlap_state, true_state, named_chunkoverlap_state], outputs = [chunk_overlap_state])  # This might not work. I'm not confident it will.
        c_overlap_slide.change(fn = change_state, inputs = [c_overlap_slide, chunk_overlap_state, true_state, named_chunkoverlap_state], outputs = [chunk_overlap_state])  # this used to update both the chunk size and overlap. Now this just updates the overlap.
        c_summary_slide.change(fn = change_state, inputs = [c_summary_slide, chunk_summary_state, true_state, name_chunksum_state], outputs = [chunk_summary_state])
        c_batch_slide.change(fn = change_state, inputs = [c_batch_slide, chunk_batches_state, true_state, name_chunkbatch_state], outputs = [chunk_batches_state])

        del_collection_btn.click(fn = delete_collection, inputs = [available_rule_systems_dd], outputs = [rule_system_state]).then(fn = find_collections, outputs = [rule_systems_list_state]).then(fn = update_drop_down, inputs = [rule_systems_list_state], outputs = [rule_systems_dd]).then(fn = update_drop_down, inputs = [rule_systems_list_state], outputs = [available_rule_systems_dd]).then(fn = find_documents, inputs = [available_rule_systems_dd], outputs = [documents_list_state]).then(fn = update_drop_down, inputs = [documents_list_state], outputs = [available_documents_dd]).then(fn = list_length, inputs = [documents_list_state], outputs = [available_documents_textbox])
        delete_document_btn.click(fn = delete_document, inputs = [available_rule_systems_dd, available_documents_dd]).then(fn = find_documents, inputs = [available_rule_systems_dd], outputs = [documents_list_state]).then(fn = update_drop_down, inputs = [documents_list_state], outputs = [available_documents_dd])
        # del_everything_btn.click(fn = delete_collection, inputs = [available_rule_systems_dd, true_state]).then(fn = find_collections, outputs = [rule_system_state]).then(fn = update_drop_down, inputs = [rule_system_state], outputs = [rule_systems_dd]).then(fn = update_drop_down, inputs = [rule_system_state], outputs = [available_documents_dd])

        embed_models_dd.select(fn = change_state, inputs = [embed_models_dd, embed_model_state, true_state, name_embed_state], outputs = [embed_model_state]).then(fn = update_textbox, inputs = [embed_model_state], outputs = [embed_textbox])

        more_metadata_btn.click(fn = update_metadata, inputs = [available_rule_systems_dd, available_documents_dd, more_metadata_dd])

        rule_system_add_btn.click(fn = change_state, inputs = [rule_systems_dd, rule_system_state, true_state, name_rule_state], outputs = [rule_system_state]).then(fn = create_collection, inputs = [rule_system_state]).then(fn = append_state_list, inputs = [rule_systems_list_state, rule_system_state], outputs = [rule_systems_list_state]).then(fn = update_drop_down, inputs = [rule_systems_list_state, rule_system_state], outputs = [rule_systems_dd]).then(fn = update_drop_down, inputs = [rule_systems_list_state, rule_system_state], outputs = [available_rule_systems_dd]).then(update_textbox, inputs = [rule_system_state], outputs = [selected_rule_system])

        rule_systems_dd.select(fn = change_state, inputs = [rule_systems_dd, rule_system_state, true_state, name_rule_state], outputs = [rule_system_state]).then(fn = update_textbox, inputs = [rule_system_state], outputs = [selected_rule_system])
        
        save_tags.click(fn = append_state_list, inputs = [tags_list_state, metadata_tags_dd], outputs = [tags_list_state]).then(fn = update_drop_down, inputs = [tags_list_state], outputs = [metadata_tags_dd]).then(fn = export_tags, inputs = [tags_list_state])

        lang_model_sum_dd.select(fn = change_state, inputs = [lang_model_sum_dd, lang_model_state, true_state, name_lang_state], outputs = [lang_model_state])

        upload_file_space.upload(fn = change_state, inputs = [rule_systems_dd], outputs = [rule_system_state]).then(fn = append_state_list, inputs = [rule_systems_list_state, rule_system_state], outputs = [rule_systems_list_state]).then(fn = update_drop_down, inputs = [rule_systems_list_state, rule_system_state], outputs = [rule_systems_dd]).then(fn = update_drop_down, inputs = [rule_systems_list_state, rule_system_state], outputs = available_rule_systems_dd).then(fn = load_documents, inputs = [upload_file_space, rule_system_state, embed_model_state, lang_model_state, metadata_tags_dd, chunk_size_state, chunk_overlap_state, chunk_batches_state, chunk_summary_state], outputs = [upload_status_box])

    return upload, {
        "chunk_batch": c_batch_slide,
        "chunk_overlap": c_overlap_slide,
        "chunk_size": c_size_slide,
        "chunk_sum": c_summary_slide,
        "embed_models_dd": embed_models_dd, 
        "embed_textbox": embed_textbox, 
        "lang_model_sum_dd": lang_model_sum_dd,
        "list_of_db_dd": list_of_db_dd, 
        "metadata_tags_dd": metadata_tags_dd, 
        "rule_systems_dd_1": available_rule_systems_dd, 
        "rule_systems_dd_2": rule_systems_dd, 
        "upload_status_box": upload_status_box
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

    TECH_UPLOAD, upload_components = create_upload(["DB Path"], ["Embedding Choice"], ["AD&D", "Shadowrun", "Rifts"], ["Tag 1", "Tag 2", "Tag 3"])
else:
    logger = logging.getLogger(__name__)
    print("Rendered Upload Tab")
    logger.info("Rendered Upload Tab @ (time to be implemented)")
