import gradio as gr


with gr.Blocks() as upload:
    with gr.Row():
        with gr.Column():
            supported_files = gr.Markdown("Supported file types: .pdf, .docx, .txt, .csv")
            upload_file = gr.File(label = f"Drag and drop a file")
        with gr.Column():
            c_size = gr.Slider(minimum = 10, maximum = 2_000, value = 512, label = "Chunk Size", interactive = True)
            c_overlap = gr.Slider(minimum = 1, maximum = 1_000, value = 50, label = "Chunk Overlap", interactive = True)
            with gr.Row():
                with gr.Column():
                    rule_systems = gr.Dropdown(label = "Rule Systems", value = 0, choices = [x for x in range(10)], interactive = True)
                with gr.Column():
                    new_rule_system = gr.Textbox(label = None, submit_btn = True, placeholder = "Type in a new rule system/collection")

    with gr.Row():
        with gr.Accordion(label = "Database of Holding", open = False):
            with gr.Row():
                with gr.Column():
                    available_collections = gr.Dropdown(choices = [x for x in range(5)], info = "Choose Collection", interactive = True)
                with gr.Column():
                    available_documents = gr.Dropdown(choices = [x for x in range(3)], info = "Available documents in selected collection", interactive = True)


if __name__ in "__main__":
    upload.launch()