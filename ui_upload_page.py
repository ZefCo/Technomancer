import gradio as gr
from RAG_pipeline import load_documents
from ui_functions import load_settings, write_rules

CHUNK_SIZE = 512
CHUNK_OVERLAP = 50
SETTINGS = load_settings()
RULE_SYSTEMS = SETTINGS["rule_systems"]
# RAG_in = RAG_input()

with gr.Blocks() as page:
    chunk_size = gr.State(value = 512)
    chunk_overlap = gr.State(value = 50)
    rule_systems = gr.State(value = RULE_SYSTEMS)
    master_settings = gr.State(value = SETTINGS)
    index_id = gr.State(value = "rule_systems")  # this is here because Gradio really wants everything to have a specific id when it's passed in the .then() later.
    collection = gr.State(value = RULE_SYSTEMS[0])

    short_about = gr.HTML("<h2>About</h2>" \
    "<p>For uploading files. Due to the nature of RPG books having lots of tables and odd graphics, you have the ability to adjust the chunk size and overlap. Chunk size refers to how the document is ''sliced up'' and stored, and chunk overlap is how those slices overlap so that if it is sliced on important information, the information is properly retrived.</p>"
    "<p>It is important, again, due to how the documents are stored, to audit what is stored after it is uploaded. Make sure the document is being read properly. If it is not, delete the document, and adjust the chunk sizes and overlap. There is a bit of trial and error here. A good rule of thumb for the chunk size and overlap: 128-256 is good for facts, 256-512 is good general purpose, and 512=1024 is good for complex technical documents."
    "<p>The Rule system acts as the database collection, how the document is stored. ChromaDB requires that collections be between 3 and 512 characters long, can contain digits, letters, dots (.), dashes (-), and underscores (_), not contain IP addresses, not have consecutive dots, and must end with a lower case letter or digit. For this reason the rule systems will be modified when passed for a collection: they will be all lower case, & turned into an n, and special characters changed to their non shifted number. In practice, this database will change the string collection into a series of ascii characters. This means your rule book names should be <120 (rounding down for safety).")
    
    # with gr.Row():
    #     gr.UploadButton("Upload File", file_types = [".pdf", ".txt", ".csv", ".md", ".markdown", ".docx"])
    
    with gr.Row():
        with gr.Column():
            upload_file = gr.File(label = "Drag and drop a file to here\nSupported file types: .pdf")
        with gr.Column():
            c_size = gr.Slider(minimum = 10, maximum = 2_000, value = CHUNK_SIZE, label = "Chunk Size", interactive = True, step = 1)
            c_overlap = gr.Slider(minimum = 0, maximum = 200, value = CHUNK_OVERLAP, label = "Chunk Overlap", interactive = True, step = 1)
    
            with gr.Row():
                with gr.Column():
                    rule_system_dd = gr.Dropdown(label = "Rule System", value = RULE_SYSTEMS[0], choices = RULE_SYSTEMS, interactive = True)
                with gr.Column():
                    new_rule_system = gr.Textbox(label = None, submit_btn = True, placeholder = "Input a new rule system or collection")
    

    gr.Button("Delete Document (not working yet)")
    warning = gr.HTML("<h2>The following is here only for testing purposes and will be removed in the future!</h2><h2>Do not press unless you mean it!</h2>")
    gr.Button("Delete Database")

    def add_new_rule_system(new_system, system_list):
        '''
        Updates the rule system drop down menu.
        '''
        if new_system and new_system not in tuple(system_list):
            system_list.append(new_system)
            system_list.sort()
        return "", gr.Dropdown(choices = system_list)
    
    def update_collection(new_collection):
        '''
        Updates and changes the collection state
        '''
        return new_collection

    new_rule_system.submit(fn = add_new_rule_system, inputs = [new_rule_system, rule_systems], outputs = [new_rule_system, rule_system_dd]).then(write_rules, inputs = [master_settings, rule_systems, index_id])
    rule_system_dd.change(fn = update_collection, inputs = rule_system_dd, outputs = collection)
    upload_file.upload(fn = load_documents, inputs = [upload_file, collection, chunk_size, chunk_overlap])

if __name__ in "__main__":
    page.launch()
