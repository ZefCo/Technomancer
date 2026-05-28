import gradio as gr
import ui_front_page, ui_chat_page, ui_about_page, ui_upload_page
# from ui_functions import find_models, load_settings, write_settings_index

# SETTINGS = load_settings()
# MODELS = find_models()
# write_settings_index(SETTINGS, MODELS, "models")  # This updates the models present in the UI every time the program is run.
#                                                   # This is only done at the very beginning because the user shouldn't be putting in new models on the fly.

# So these things only load once, but they do slow down the overall starting of the server. That might not be a problem, as it could just take a little while to get the thing
# up and running once, and then we're OK. But, is there a way we can load the settings in the start page once and then pass them to the needed pages?

# https://www.gradio.app/guides/state-in-blocks This is how you can pass information from one thing to another. With this in mind, there might be a way load up the needed functions
# and settings here, once, and pass them to all the other pages as needed. Browser state is also a way to keep the chat going.


with gr.Blocks(title = "Technomancer v0.1") as page:
    navbar = gr.Navbar(visible = False)
    ui_front_page.page.render()

with page.route("Chatbot"):
    # navbar = gr.Navbar(visible = True, main_page_name = "Chat Page")  #, value = [("Documentation", "https://docs.example.com")])
    ui_chat_page.page.render()

with page.route("About"):
    ui_about_page.page.render()

with page.route("Upload"):
    ui_upload_page.page.render()

# with page.route("Other"):
#     ui_other_page.page.render()

if __name__ in "__main__":
    page.launch(server_name = "0.0.0.0", server_port = 7860)