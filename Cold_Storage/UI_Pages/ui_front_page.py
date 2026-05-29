import gradio as gr
import socket

IP: str = socket.gethostbyname(socket.gethostname())

print("Finished Loading Front Page")


with gr.Blocks() as page:
    title = gr.HTML("<h1>Technomancer</h1>")
    gr.Button(value = "Enter Chatbot", link = f"/chatbot")  # will not work with the local computer. This forces it to have a weird IP like 127.0.0.1:IP Address:7860
    gr.Button(value = "About/Manual", link = f"/about")     # Or I could just ignore the buttons, and use the navigator
    gr.Button(value = "Upload Document", link = f"/upload")
    # gr.Button(value = "Other Tools", link = "http://127.0.0.1:7860/other")

if __name__ in "__main__":
    page.launch()