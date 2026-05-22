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
        with gr.Column():
            logout_btn = gr.Button(value = "Logout")


if __name__ in "__main__":
    interface.launch()