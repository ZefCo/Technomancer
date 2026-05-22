import gradio as gr
import ui_front_page, ui_chat_page


with gr.Blocks() as pages:
    navbar = gr.Navbar(visible = True, main_page_name = "Login Page")  #, value = [("About", "https://example.com/about")])
    ui_front_page.pages.render()

with pages.route("Chat"):
    navbar = gr.Navbar(visible = True, main_page_name = "Chat Page")  #, value = [("Documentation", "https://docs.example.com")])
    # ui_chat_page.pages.render()

if __name__ in "__main__":
    pages.launch()