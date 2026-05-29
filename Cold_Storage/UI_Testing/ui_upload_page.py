import gradio as gr
import os
import shutil

with gr.Blocks() as page:
    short_about = gr.HTML("<h2>Aboute</h2>" \
    "<p>For uploading files. Due to the nature of RPG books having lots of tables and odd graphics, you have the ability to adjust the chunk size and overlap. Chunk size refers to how the document is ''sliced up'' and stored, and chunk overlap is how those slices overlap so that if it is sliced on important information, the information is properly retrived.</p>"
    "<p>It is important, again, due to how the documents are stored, to audit what is stored after it is uploaded. Make sure the document is being read properly. If it is not, delete the document, and adjust the chunk sizes and overlap. There is a bit of trial and error here.")
    
    with gr.Row():
        gr.Button("Upload File (not working yet)")
    with gr.Row():
        with gr.Column():
            gr.Slider(minimum = 10, maximum = 10_000, value = 4_000, label = "Chunk Size", interactive = True)
        with gr.Column():
            gr.Slider(minimum = 0, maximum = 2_000, value = 200, label = "Chunk Overlap", interactive = True)

    gr.Button("Delete Document (not working yet)")
    warning = gr.HTML("<h2>The following is here only for testing purposes and will be removed in the future!</h2><h2>Do not press unless you mean it!</h2>")
    gr.Button("Delete Database")