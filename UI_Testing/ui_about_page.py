import gradio as gr


with gr.Blocks() as page:
    title = gr.HTML("<h1>About</h1>")
    body = gr.HTML("For details and instructions about how to use this.")



if __name__ in "__main__":
    page.launch()