import gradio as gr


with gr.Blocks() as splash:
    title = gr.HTML("<h1>Technomancer</h1>")
    gr.Button(value = "Enter")

if __name__ in "__main__":
    splash.launch()