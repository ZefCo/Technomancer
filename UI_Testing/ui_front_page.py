import gradio as gr


with gr.Blocks() as page:
    title = gr.HTML("<h1>Technomancer</h1>")
    gr.Button(value = "Enter Chatbot", link = "http://127.0.0.1:7860/chatbot")
    gr.Button(value = "About/Manual", link = "http://127.0.0.1:7860/about")
    gr.Button(value = "Upload Document", link = "http://127.0.0.1:7860/upload")
    # gr.Button(value = "Other Tools", link = "http://127.0.0.1:7860/other")

if __name__ in "__main__":
    page.launch()