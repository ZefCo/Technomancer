import gradio as gr


with gr.Blocks() as page:
    title = gr.HTML("<h1>Technomancer</h1>")
    gr.Button(value = "Enter", link = "http://127.0.0.1:7860/chatbot")
    gr.Button(value = "Other Tools", link = "http://127.0.0.1:7860/other")
    gr.Button(value = "Upload Document")
    gr.Button(value = "About/Manual", link = "http://127.0.0.1:7860/about")

if __name__ in "__main__":
    page.launch()