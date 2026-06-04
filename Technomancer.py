import pathlib
from os.path import basename
from __log_fn import setup_logs
from datetime import datetime
import logging

cwd = pathlib.Path.cwd()
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
log_dir = cwd / "Logs"
log_dir.mkdir(parents = True, exist_ok = True)
    
# logger = logging.getLogger(__name__)
    
logger = setup_logs(log_dir / f"Technomancer__{timestamp}.log", level = logging.INFO)
# logger_debug = setup_logs(log_dir / f"Technomancer_DEBUG_{timestamp}.log")


import gradio as gr

from __rag_pipeline import find_collections, find_documents

import __tech_about as tech_about
from __tech_chat import create_chat
from __tech_fn import sort_models, load_paths, load_tags, update_drop_down, update_system_prompt, update_textbox, import_setting, chatbot_avatars
from __tech_upload import create_upload


def launch_technomancer():
    cwd = pathlib.Path.cwd()

    RULE_SYSTEMS = find_collections()  # this will be the various collections in the database.
    LANG_MODELS, EMBED_MODELS = sort_models()
    DBOH = load_paths()
    TAGS = load_tags()
    IMAGES = import_setting(cwd / "Settings" / "AvatarImages.yaml")


    with gr.Blocks(title = "Technomancer v0.5") as Technomancer:
        # A series of state variables. They are loaded here and shared among the tabs, which allows one state to be modified in one tab and then updated in another.
        db_paths = gr.State(value = DBOH)
        rule_systems = gr.State(value = RULE_SYSTEMS)
        lang_models = gr.State(value = LANG_MODELS)
        embed_models = gr.State(value = EMBED_MODELS)
        tags = gr.State(value = TAGS)
        user_avatar = gr.State(value = IMAGES["user"])
        bot_avatar = gr.State(value = IMAGES["bot"])

        # because of how many state variables I'm juggeling, I'm going to keep them alphabitized.
        with gr.Tabs():
            with gr.Tab(label = "About/Manual"):
                tech_about.about.render()
            
            with gr.Tab(label = "Technomancer Chat") as chat_tab:
                try:
                    TECH_CHAT, chat_components = create_chat(embed_models, lang_models, rule_systems)
                except Exception as e:
                    # log that chat cannot be loaded properly.
                    logger.critical(f"Something went wrong when starting Chat Tab: Error type {type(e)}")
                    logger.critical(f"{e}")
                    raise RuntimeError("Cannot load Chat Tab")


            with gr.Tab(label = "Database of Holding") as upload_tab:
                try:
                    TECH_UPLOAD, upload_components = create_upload(db_paths, embed_models, rule_systems, tags)
                except Exception as e:
                    # loga that upload tab cannot be loaded properly.
                    logger.critical(f"Something went wrong when starting Upload Tab: Error type {type(e)}")
                    logger.critical(f"{e}")
                    raise RuntimeError("Cannot load Upload Tab")

        upload_tab.select(fn = update_drop_down, inputs = [rule_systems], outputs = [upload_components["available_collections"]]).then(fn = update_drop_down, inputs = [rule_systems], outputs = [upload_components["rule_system"]]).then(fn = update_drop_down, inputs = [embed_models], outputs = [upload_components["eb_model"]]).then(fn = update_drop_down, inputs = [db_paths], outputs = [upload_components["list_of_db"]])
        chat_tab.select(fn = update_drop_down, inputs = [lang_models], outputs = [chat_components["model_choice"]]).then(fn = update_drop_down, inputs = [rule_systems], outputs = [chat_components["collection_choice"]]).then(fn = update_drop_down, inputs = [embed_models], outputs = [chat_components["embedding_choice"]]).then(fn = chatbot_avatars, inputs = [user_avatar, bot_avatar], outputs = [chat_components["chatbot"]])

    return Technomancer


if __name__ in "__main__":
    Technomancer = launch_technomancer()
    Technomancer.launch(server_name = "0.0.0.0", server_port = 7860)