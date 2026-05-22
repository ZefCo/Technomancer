import gradio as gr
import ui_front_page, ui_chat_page, ui_about_page, ui_other_page


with gr.Blocks() as page:
    navbar = gr.Navbar(visible = False)
    ui_front_page.page.render()

with page.route("Chatbot"):
    # navbar = gr.Navbar(visible = True, main_page_name = "Chat Page")  #, value = [("Documentation", "https://docs.example.com")])
    ui_chat_page.page.render()

with page.route("About"):
    ui_about_page.page.render()

with page.route("Other"):
    ui_other_page.page.render()

if __name__ in "__main__":
    page.launch()