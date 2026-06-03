from datetime import datetime
import pathlib
from os.path import basename
from __log_fn import setup_logs
import logging
logger = logging.getLogger(__name__)
setup_logs(pathlib.Path(basename(__file__)).stem)

start = datetime.now()
logger.info(f"Started loading Chat Tab script | Start {start.strftime("%H:%M:%S")}")

import gradio as gr

from __rag_pipeline import load_documents, find_documents, delete_document, create_collection

from __tech_fn import update_drop_down, append_state_list, update_textbox, change_state, change_state_list, update_slider, change_state_per



def create_upload(db_paths, embed_models, rule_systems, tags):
    '''
    '''
    chunk_overlap_s = gr.State(value = 50)
    chunk_size_s = gr.State(value = 512)
    default_text_message_s = gr.State(value = "Type in a new rule system/collection")
    documents_listed_s = gr.State([])
    document_tags_s = gr.State(value = [])
    empty_values_s = gr.State(value = [])  # this is just to clear a drop down
    false_state_s = gr.State(value = False)
    overlap_name_s = gr.State(value = "Chunk Overlap")
    percent_s = gr.State(value = 0.1)
    previous_embed_s = gr.State(value = None)
    previous_chunk_overlap_s = gr.State(value = None)
    previous_chunk_size_s = gr.State(value = None)
    size_name_s = gr.State(value = "Chunk Size")
    true_state_s = gr.State(value = True)

    
    with gr.Blocks() as upload:
        
        with gr.Row(equal_height = True):
            
            with gr.Column(variant = "panel"):
                gr.Markdown("Supported file types: .pdf, .docx, .txt, .csv, .epub")
                upload_file_space = gr.File(label = f"Drag and drop a file")
            
            with gr.Column(variant = "panel"):
            
                # with gr.Row():
                gr.Markdown("## Rule System & Tags")
                gr.Markdown("All documents need to be added to a rule system, a group that the document can belong to. To create a rule system group, select from or type into the dropdown menu. Tags can be added to any document, which will make searching those documents with Technomancer easier. If none is selected, the document will be added to 'Generic.'")
                rule_systems_dd = gr.Dropdown(label = "Rule System to Upload to (Required)", choices = [], interactive = True, allow_custom_value = True)
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
                
        c_size_slide.change(fn = change_state, inputs = [c_size_slide, previous_chunk_size_s, true_state_s, size_name_s], outputs = [chunk_size_s]).then(fn = update_slider, inputs = [chunk_size_s, percent_s], outputs = [c_overlap_slide]).then(fn = change_state, inputs = [c_overlap_slide, previous_chunk_overlap_s, true_state_s, overlap_name_s], outputs = [chunk_overlap_s])  # This might not work. I'm not confident it will.
        c_overlap_slide.change(fn = change_state, inputs = [c_overlap_slide, previous_chunk_overlap_s, true_state_s, overlap_name_s], outputs = [chunk_overlap_s])  # this used to update both the chunk size and overlap. Now this just updates the overlap.

        # Look into how to log this information.
        available_rule_systems_dd.select(fn = find_documents, inputs = [available_rule_systems_dd], outputs = [documents_listed_s]).then(fn = update_drop_down, inputs = [documents_listed_s], outputs = [available_documents_dd])
        
        upload_file_space.upload(fn = load_documents, inputs = [upload_file_space, rule_systems_dd, chunk_size_s, chunk_overlap_s, eb_model_dd]).then(fn = append_state_list, inputs = [rule_systems, rule_systems_dd], outputs = [rule_systems])

        delete_document_btn.click(fn = delete_document, inputs = [available_rule_systems_dd, available_documents_dd]).then(fn = find_documents, inputs = [available_rule_systems_dd], outputs = [documents_listed_s]).then(fn = update_drop_down, inputs = [documents_listed_s], outputs = [available_documents_dd])
        
        add_doc_tags.click(fn = change_state_list, inputs = [document_tags_dd], outputs = [document_tags_s])

    return upload, {"available_collections": available_rule_systems_dd, "rule_system": rule_systems_dd, "eb_model": eb_model_dd, "list_of_db": list_of_db_dd}


if __name__ in "__main__":
    TECH_UPLOAD, upload_components = create_upload(["DB Path"], ["Embedding Choice"], ["AD&D", "Shadowrun", "Rifts"], ["Tag 1", "Tag 2", "Tag 3"])
else:
    end = datetime.now()
    delta = end - start
    logger.info(f"Finished loading Upload Tab script | End {end.strftime("%H:%M:%S")} | Total load time: {delta.microseconds} microseconds (10**-6 s)")
    print("Rendered Upload Tab")