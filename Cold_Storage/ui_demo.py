# http://127.0.0.1:7860/

# https://www.gradio.app/
# https://www.gradio.app/guides/creating-a-custom-chatbot-with-blocks
# https://www.gradio.app/docs/gradio/button
# https://www.gradio.app/docs/gradio/column
# https://www.gradio.app/guides/controlling-layout
# https://loveaiblog.github.io/2025/03/02/Designing-UI-with-Gradio/

import gradio as gr

with gr.Blocks() as interface:
    with gr.Row():
        chatbot = gr.Chatbot()

    with gr.Row(variant = "compact"):
        with gr.Column(scale = 12):
            msg = gr.Textbox(show_label = True)
        with gr.Column(scale = 1):
            with gr.Row():
                submit_btn = gr.Button(value = "Submit")
            with gr.Row():
                clear_btn = gr.ClearButton()
    
    with gr.Row():
        with gr.Column():
            upload_btn = gr.Button(value = "Upload File")
        with gr.Column():
            blank_btn = gr.Button(value = "Nothing")



if __name__ in "__main__":
    interface.launch()