import gradio as gr
import pathlib
from __rag_pipeline import find_collections, find_documents
import __tech_about as tech_about
from __tech_chat import create_chat
from __tech_fn import sort_models, load_paths, load_tags, update_drop_down, update_system_prompt, update_textbox, update_chunks
from __tech_upload import create_upload

cwd = pathlib.Path.cwd()
SYSTEM_CONTENT = "You are a DM for tabletop RPGs named Technomancer and are friendly. Assume the user already knows a lot of the terminology. You are not meant to generate new campaign ideas, rules, but are meant to help reference rules, tables, pages, NPCs, and the like."
RULE_SYSTEMS = find_collections()  # this will be the various collections in the database.
LANG_MODELS, EMBED_MODELS = sort_models()
# DBoH = cwd / "DB_of_Holding"
DBOH = load_paths()
TAGS = load_tags()


with gr.Blocks(title = "Technomancer v0.5") as Technomancer:
    # A series of state variables. They are loaded here and shared among the tabs, which allows one state to be modified in one tab and then updated in another.
    db_paths = gr.State(value = DBOH)
    system_content = gr.State(value = SYSTEM_CONTENT)
    rule_systems = gr.State(value = RULE_SYSTEMS)
    lang_models = gr.State(value = LANG_MODELS)
    embed_models = gr.State(value = EMBED_MODELS)
    tags = gr.State(value = TAGS)

    # because of how many state variables I'm juggeling, I'm going to keep them alphabitized.
    with gr.Tabs():
        with gr.Tab(label = "About/Manual"):
            tech_about.about.render()
         
        with gr.Tab(label = "Technomancer Chat") as chat_tab:
            # tech_chat.chat.render()
            TECH_CHAT, chat_components = create_chat(lang_models, rule_systems, system_content, embed_models)

        with gr.Tab(label = "Database of Holding") as upload_tab:
            TECH_UPLOAD, upload_components = create_upload(db_paths, embed_models, rule_systems, tags)

    # upload_tab.select(fn = update_drop_down, inputs = [], outputs = [])
    upload_tab.select(fn = update_drop_down, inputs = [rule_systems], outputs = [upload_components["available_collections"]]).then(update_drop_down, [rule_systems], [upload_components["rule_system"]]).then(fn = update_drop_down, inputs = [embed_models], outputs = [upload_components["eb_model"]]).then(fn = update_drop_down, inputs = [db_paths], outputs = [upload_components["list_of_db"]])
    chat_tab.select(fn = update_drop_down, inputs = [lang_models], outputs = [chat_components["model_choice"]]).then(fn = update_drop_down, inputs = [rule_systems], outputs = [chat_components["collection_choice"]]).then(fn = update_drop_down, inputs = [embed_models], outputs = [chat_components["embedding_choice"]])
    


if __name__ in "__main__":
    Technomancer.launch(server_name = "0.0.0.0", server_port = 7860)