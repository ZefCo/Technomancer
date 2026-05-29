import gradio as gr
import tech_about, tech_chat, tech_upload


with gr.Blocks(title = "Technomancer v0.1") as Technomancer:
    with gr.Tabs():
        with gr.Tab(label = "About/Manual"):
            tech_about.about.render()
         
        with gr.Tab(label = "Technomancer Chat"):
            tech_chat.chat.render()

        with gr.Tab(label = "Database of Holding"):
            tech_upload.upload.render()


if __name__ in "__main__":
    Technomancer.launch()