from os.path import basename
from __log_fn import setup_logs
import logging

import gradio as gr

from __rag_pipeline import load_documents, find_documents, delete_document, create_collection

from __states import chunk_batches_state, chunk_overlap_state, chunk_size_state, documents_list_state, embed_model_state, named_chunkoverlap_state, name_chunksize_state, name_embed_state, name_rule_state, percent_state, rule_system_state, rule_systems_list_state, tags_list_state, true_state, upload_status_state

from __tech_fn import update_drop_down, append_state_list, change_state, change_state_list, update_slider, update_textbox


def create_upload():
    '''
    '''
    
    with gr.Blocks() as upload:
        
        with gr.Row(equal_height = True):
                        
            with gr.Column(variant = "panel"):
            
                # with gr.Row():
                gr.Markdown("## Rule System & Tags")
                gr.Markdown("All documents need to be added to a rule system, a group that the document can belong to. To create a rule system group, select from or type into the dropdown menu. Tags can be added to any document, which will make searching those documents with Technomancer easier. If none is selected, the document will be added to 'Generic.'")
                with gr.Column(variant = "panel"):
                    with gr.Row():
                        rule_systems_dd = gr.Dropdown(label = "Rule System to Upload to (Required)", choices = [], interactive = True, allow_custom_value = True, scale = 3)
                        rule_system_add_btn = gr.Button(value = "Add Rule System", scale = 1)

                    # new_rule_system = gr.Textbox(label = None, submit_btn = True, placeholder = "Type in a new rule system/collection")
                
                with gr.Row():
                    document_tags_dd = gr.Dropdown(choices = ["NPCs", "Lore", "Optional", "Sci Fi", "Fantasy"], label = "Tags to organize documents (not implemented yet)", multiselect = True, allow_custom_value = True, scale = 2)
                    add_doc_tags = gr.Button(value = "Add Tags", scale = 1, size = "lg")

            with gr.Column(variant = "panel"):
                gr.Markdown("While this can go as high as 4,000 chunks, consider staying within 128-1024, with overlap being about 0.10 - 0.20 of the chunk size. Chunk batches is for how many chunks will be added at a time. Due to the size of the books, there can be several thousand chunks that are needed to be loaded.")
                c_size_slide = gr.Slider(minimum = 10, maximum = 4_000, value = 512, label = "Chunk Size", interactive = True)
                c_overlap_slide = gr.Slider(minimum = 1, maximum = 2_000, value = 50, label = "Chunk Overlap", interactive = True)
                c_batch_slide = gr.Slider(minimum = 1, maximum = 4_000, value = 50, label = "Chunk Batches", interactive = True)

            with gr.Column(variant = "panel"):
                gr.Markdown("Supported file types: .pdf, .docx, .txt, .csv, .epub")
                selected_rule_system = gr.Textbox(label = "Add to rule system", interactive = False, value = "Select Rule system")
                upload_file_space = gr.File(label = f"Drag and drop a file")
                upload_status_box = gr.Textbox(label  = "Upload Status", interactive = False, value = "No file uploaded yet")

        with gr.Row():

            with gr.Accordion(label = "Database of Holding Management", open = False) as DBoH_acc:

                with gr.Accordion(label = "Delete Documents", open = False) as del_DBoH_acc:

                    with gr.Row():
                        gr.Markdown("# Note on Deletion:\nThis deletes all things based on the file name of the book. If the book name is close enough to [an]other book[s], it is possible that the other book[s] will be delete too! Check to make sure that the books you want to stay in the database are in fact still there!")

                    with gr.Row():
                        with gr.Column():
                            available_rule_systems_dd = gr.Dropdown(choices = [], label = "Choose Collection", interactive = True)

                        with gr.Column():
                            available_documents_dd = gr.Dropdown(choices = [], label = "Available documents in selected collection", interactive = True)

                    with gr.Row():
                        clear_collection_btn = gr.Button(value = "Removes Rule System (not yet implemented)")
                        delete_document_btn = gr.Button(value = "Delete Selected Document")
                    
                
                with gr.Accordion(label = "Advanced Settings", open = False) as adv_DBoH_acc:
                
                    with gr.Row():
                        with gr.Column():
                            gr.Markdown("Embedding Model choice. If you have multiple databases, this can be used with different embedding models and switch between the two. Not fully implemented yet.")
                            eb_model_dd = gr.Dropdown(label = "Embedding Models", choices = [], info = "Choice of embedding model. Only for advanced uses.", interactive = True)

                        with gr.Column():
                            gr.Markdown("Note this has not be implemented yet as I need to figure out how to pass all this information to and from Gradio. Will use Settings files probably.")
                            list_of_db_dd = gr.Dropdown(info =  "Not implemented yet", label = "Choose Database", choices = [], interactive = True)
                            db_path_space = gr.Markdown("Reserved for Create Path to Database option. (Not implemented yet. Button? Textbox? Dropdowns? Not sure what to do here.)")
                        
                        with gr.Column():
                            gr.Markdown("Reserved for other options. Not sure what they are.")
                
        # add_doc_tags.click(fn = change_state_list, inputs = [document_tags_dd], outputs = [document_tags_s])
        add_doc_tags.click(fn = change_state_list, inputs = [document_tags_dd], outputs = [tags_list_state])

        available_rule_systems_dd.select(fn = find_documents, inputs = [available_rule_systems_dd], outputs = [documents_list_state]).then(fn = update_drop_down, inputs = [documents_list_state], outputs = [available_documents_dd])

        c_size_slide.change(fn = change_state, inputs = [c_size_slide, chunk_size_state, true_state, name_chunksize_state], outputs = [chunk_size_state]).then(fn = update_slider, inputs = [chunk_size_state, percent_state], outputs = [c_overlap_slide]).then(fn = change_state, inputs = [c_overlap_slide, chunk_overlap_state, true_state, named_chunkoverlap_state], outputs = [chunk_overlap_state])  # This might not work. I'm not confident it will.
        c_overlap_slide.change(fn = change_state, inputs = [c_overlap_slide, chunk_overlap_state, true_state, named_chunkoverlap_state], outputs = [chunk_overlap_state])  # this used to update both the chunk size and overlap. Now this just updates the overlap.

        delete_document_btn.click(fn = delete_document, inputs = [available_rule_systems_dd, available_documents_dd]).then(fn = find_documents, inputs = [available_rule_systems_dd], outputs = [documents_list_state]).then(fn = update_drop_down, inputs = [documents_list_state], outputs = [available_documents_dd])
        
        eb_model_dd.select(fn = change_state, inputs = [eb_model_dd, embed_model_state, true_state, name_embed_state], outputs = [embed_model_state])

        rule_system_add_btn.click(fn = create_collection, inputs = [rule_systems_dd]).then(fn = append_state_list, inputs = [rule_systems_list_state, rule_systems_dd], outputs = [rule_systems_list_state]).then(fn = update_drop_down, inputs = [rule_systems_list_state], outputs = [rule_systems_dd]).then(fn = update_drop_down, inputs = [rule_systems_list_state], outputs = [available_rule_systems_dd])

        rule_systems_dd.select(fn = change_state, inputs = [rule_systems_dd, rule_system_state, true_state, name_rule_state], outputs = [rule_system_state]).then(fn = update_textbox, inputs = [rule_system_state], outputs = [selected_rule_system])

        upload_file_space.upload(fn = append_state_list, inputs = [rule_systems_list_state, rule_systems_dd], outputs = [rule_systems_list_state]).then(fn = change_state_list, inputs = [rule_systems_list_state], outputs = [rule_systems_dd]).then(fn = change_state_list, inputs = [rule_systems_list_state], outputs = available_rule_systems_dd).then(fn = load_documents, inputs = [upload_file_space, rule_system_state, chunk_size_state, chunk_overlap_state, eb_model_dd], outputs = [upload_status_box])

    return upload, {"embed_models_dd": eb_model_dd, "list_of_db_dd": list_of_db_dd, "rule_systems_dd_1": available_rule_systems_dd, "rule_systems_dd_2": rule_systems_dd, "upload_status_box": upload_status_box}
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
