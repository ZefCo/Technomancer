import gradio as gr
import pathlib
from __rag_pipeline import find_collections, find_documents
import __tech_about as tech_about
from __tech_chat import create_chat
from __tech_fn import sort_models, load_paths, load_tags
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
         
        with gr.Tab(label = "Technomancer Chat"):
            # tech_chat.chat.render()
            TECH_CHAT = create_chat(lang_models, rule_systems, system_content)

        with gr.Tab(label = "Database of Holding"):
            TECH_UPLOAD = create_upload(db_paths, embed_models, rule_systems, tags)


if __name__ in "__main__":
    Technomancer.launch(server_name = "0.0.0.0", server_port = 7860)