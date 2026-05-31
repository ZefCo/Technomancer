import gradio as gr
from __rag_pipeline import load_documents, find_documents, delete_document, create_collection
from __tech_fn import update_drop_down, append_state_list, update_textbox, update_chunks, change_state_list


def create_upload(dboh_paths, models, rule_systems, tags):
    '''
    '''
    documents_listed = gr.State([])
    default_text_message = gr.State(value = "Type in a new rule system/collection")
    chunk_size = gr.State(value = 512)
    chunk_overlap = gr.State(value = 50)
    document_tags = gr.State(value = [])
    empty_values = gr.State(value = [])  # this is just to clear a drop down
    
    with gr.Blocks() as upload:
        
        with gr.Row():
            
            with gr.Column():
                gr.Markdown("Supported file types: .pdf, .docx, .txt, .csv")
                upload_file = gr.File(label = f"Drag and drop a file")
            
            with gr.Column():
            
                # with gr.Row():
                gr.Markdown("## Rule System & Tags")
                gr.Markdown("All documents need to be added to a rule system, just a group that the document can belong to. Tags can be added to any document, which will make searching those documents with Technomancer easier.")
                rule_system = gr.Dropdown(label = "Rule System to Upload to (Required)", choices = [], interactive = True, allow_custom_value = True)
                    # new_rule_system = gr.Textbox(label = None, submit_btn = True, placeholder = "Type in a new rule system/collection")
                with gr.Row():
                    document_tags_dd = gr.Dropdown(choices = ["NPCs", "Lore", "Optional", "Sci Fi", "Fantasy"], label = "Tags to organize documents (not implemented yet)", multiselect = True, allow_custom_value = True, scale = 2)
                    add_doc_tags = gr.Button(value = "Add Tags", scale = 1)

            with gr.Column():
                gr.Markdown("While this can go as high as 4,000 chunks, consider staying within 128-1024, with overlap being about 0.10 - 0.20 of the chunk size")
                c_size = gr.Slider(minimum = 10, maximum = 4_000, value = 512, label = "Chunk Size", interactive = True)
                c_overlap = gr.Slider(minimum = 1, maximum = 2_000, value = 50, label = "Chunk Overlap", interactive = True)

        with gr.Row():

            with gr.Accordion(label = "Database of Holding", open = False) as DBoH:

                with gr.Row():

                    with gr.Column():
                        available_collections = gr.Dropdown(choices = [], label = "Choose Collection", interactive = True)

                    with gr.Column():
                        available_documents = gr.Dropdown(choices = [], label = "Available documents in selected collection", interactive = True)

                with gr.Row():
                    clear_collection_btn = gr.Button(value = "Removes Rule System (not yet implemented)")
                    delete_document_btn = gr.Button(value = "Delete Selected Document")
                
                with gr.Row():
                    gr.Markdown("# Note on Deletion:\nThis deletes all things based on the file name of the book. If the book name is close enough to [an]other book[s], it is possible that the other book[s] will be delete too! Check to make sure that the books you want to stay in the database are in fact still there!")
                
                with gr.Accordion(label = "Advanced Settings", open = False) as adv_DBoH:
                
                    with gr.Row():
                
                        with gr.Column():
                            eb_model = gr.Dropdown(label = "Embedding Models", choices = [], info = "Choice of embedding model. Only for advanced uses.", interactive = True)
                
                        with gr.Column():
                            gr.Textbox("Note this has not be implemented yet as I need to figure out how to pass all this information to and from Gradio. Will use Settings files probably.")
                            list_of_db = gr.Dropdown(label = "Choose Database", choices = [], info =  "Not implemented yet")
                            db_path = gr.Textbox(label = "Create path to Database. (Not implemented yet. Button? Textbox? Dropdowns? Not sure what to do here.)")
        
        DBoH.expand(fn = update_drop_down, inputs = [rule_systems], outputs = [available_collections]).then(update_drop_down, [rule_systems], [rule_system])
        DBoH.collapse(fn = update_drop_down, inputs = [rule_systems], outputs = [available_collections]).then(update_drop_down, [rule_systems], [rule_system])
        
        # new_rule_system.submit(fn = update_state_list, inputs = [rule_systems, new_rule_system], outputs = [rule_systems]).then(fn = create_collection, inputs = [new_rule_system]).then(update_textbox, [default_text_message], [new_rule_system]).then(update_drop_down, inputs = [rule_systems], outputs = [rule_system]).then(update_drop_down, inputs = [rule_systems], outputs = [available_collections]).then(fn = find_documents, inputs = [available_collections], outputs = [documents_listed]).then(fn = update_drop_down, inputs = [documents_listed], outputs = [available_documents])
        
        c_size.change(update_chunks, [c_size, c_overlap], [chunk_size, chunk_overlap])
        c_overlap.change(update_chunks, [c_size, c_overlap], [chunk_size, chunk_overlap])

        available_collections.select(fn = find_documents, inputs = [available_collections], outputs = [documents_listed]).then(fn = update_drop_down, inputs = [documents_listed], outputs = [available_documents])
        
        upload_file.upload(fn = load_documents, inputs = [upload_file, rule_system, chunk_size, chunk_overlap])

        delete_document_btn.click(fn = delete_document, inputs = [available_collections, available_documents]).then(fn = find_documents, inputs = [available_collections], outputs = [documents_listed]).then(fn = update_drop_down, inputs = [documents_listed], outputs = [available_documents])
        
        add_doc_tags.click(fn = change_state_list, inputs = [document_tags_dd], outputs = [document_tags])

        upload.load(fn = update_drop_down, inputs = [rule_systems], outputs = [available_collections]).then(update_drop_down, [rule_systems], [rule_system]).then(fn = update_drop_down, input = [models], outputs = [eb_model]).then(fn = update_drop_down, inputs = [dboh_paths], outputs = [list_of_db])


    return upload  

print("Rendered Upload tab")

# if __name__ in "__main__":
#     upload.launch()