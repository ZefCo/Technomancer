import logging
logger = logging.getLogger(__name__)

import gradio as gr

from __rag_pipeline import load_documents, find_documents, delete_document, create_collection

from __tech_fn import update_drop_down, append_state_list, update_textbox, update_chunk, change_state_list, update_slider



def create_upload(db_paths, embed_models, rule_systems, tags):
    '''
    '''
    documents_listed = gr.State([])
    default_text_message = gr.State(value = "Type in a new rule system/collection")
    chunk_size = gr.State(value = 512)
    chunk_overlap = gr.State(value = 50)
    document_tags = gr.State(value = [])
    empty_values = gr.State(value = [])  # this is just to clear a drop down
    
    with gr.Blocks() as upload:
        
        with gr.Row(equal_height = True):
            
            with gr.Column(variant = "panel"):
                gr.Markdown("Supported file types: .pdf, .docx, .txt, .csv, .epub")
                upload_file_space = gr.File(label = f"Drag and drop a file")
            
            with gr.Column(variant = "panel"):
            
                # with gr.Row():
                gr.Markdown("## Rule System & Tags")
                gr.Markdown("All documents need to be added to a rule system, a group that the document can belong to. To create a rule system group, select from or type into the dropdown menu. Tags can be added to any document, which will make searching those documents with Technomancer easier. If none is selected, the document will be added to 'Generic.'")
                rule_system_dd = gr.Dropdown(label = "Rule System to Upload to (Required)", choices = [], interactive = True, allow_custom_value = True)
                    # new_rule_system = gr.Textbox(label = None, submit_btn = True, placeholder = "Type in a new rule system/collection")
                with gr.Row():
                    document_tags_dd = gr.Dropdown(choices = ["NPCs", "Lore", "Optional", "Sci Fi", "Fantasy"], label = "Tags to organize documents (not implemented yet)", multiselect = True, allow_custom_value = True, scale = 2)
                    add_doc_tags = gr.Button(value = "Add Tags", scale = 1, size = "lg")

            with gr.Column(variant = "panel"):
                gr.Markdown("While this can go as high as 4,000 chunks, consider staying within 128-1024, with overlap being about 0.10 - 0.20 of the chunk size")
                c_size_slide = gr.Slider(minimum = 10, maximum = 4_000, value = 512, label = "Chunk Size", interactive = True)
                c_overlap_slide = gr.Slider(minimum = 1, maximum = 2_000, value = 50, label = "Chunk Overlap", interactive = True)

        with gr.Row():

            with gr.Accordion(label = "Database of Holding Management", open = False) as DBoH_acc:

                with gr.Accordion(label = "Delete Documents", open = False) as del_DBoH_acc:

                    with gr.Row():
                        gr.Markdown("# Note on Deletion:\nThis deletes all things based on the file name of the book. If the book name is close enough to [an]other book[s], it is possible that the other book[s] will be delete too! Check to make sure that the books you want to stay in the database are in fact still there!")

                    with gr.Row():
                        with gr.Column():
                            available_collections_dd = gr.Dropdown(choices = [], label = "Choose Collection", interactive = True)

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
                
        c_size_slide.change(fn = update_chunk, inputs = [c_size_slide], outputs = [chunk_size]).then(fn = update_slider, inputs = [chunk_size * .1], outputs = [c_overlap_slide]).then(fn = update_chunk, inputs = [c_overlap_slide], outputs = [chunk_overlap])  # This might not work. I'm not confident it will.
        c_overlap_slide.change(fn = update_chunk, inputs = [c_overlap_slide], outputs = [chunk_overlap]).then(fn = update_chunk, inputs = [c_size_slide], outputs = [chunk_size])  # both will update each other, just in case something has been adjusted. It's always listening for changes.

        available_collections_dd.select(fn = find_documents, inputs = [available_collections_dd], outputs = [documents_listed]).then(fn = update_drop_down, inputs = [documents_listed], outputs = [available_documents_dd])
        
        upload_file_space.upload(fn = load_documents, inputs = [upload_file_space, rule_system_dd, chunk_size, chunk_overlap, eb_model_dd]).then(fn = append_state_list, inputes = [rule_system_dd], outputs = [rule_systems])

        delete_document_btn.click(fn = delete_document, inputs = [available_collections_dd, available_documents_dd]).then(fn = find_documents, inputs = [available_collections_dd], outputs = [documents_listed]).then(fn = update_drop_down, inputs = [documents_listed], outputs = [available_documents_dd])
        
        add_doc_tags.click(fn = change_state_list, inputs = [document_tags_dd], outputs = [document_tags])

    return upload, {"available_collections": available_collections_dd, "rule_system": rule_system_dd, "eb_model": eb_model_dd, "list_of_db": list_of_db_dd}

print("Rendered Upload Tab")

if __name__ in "__main__":
    TECH_UPLOAD, upload_components = create_upload(["DB Path"], ["Embedding Choice"], ["AD&D", "Shadowrun", "Rifts"], ["Tag 1", "Tag 2", "Tag 3"])
