from datetime import datetime
import pathlib
import logging
cwd = pathlib.Path.cwd()

def setup_logs():
    '''
    Makes sure the logs are setup and ready to go
    '''
    log_dir = cwd / "Logs"
    log_dir.mkdir(exist_ok = True)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_file = log_dir / f"Technomancer_{timestamp}.log"

    logging.basicConfig(level = logging.INFO, 
                        format = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s", 
                        handlers = [logging.FileHandler(log_file)])

setup_logs()
logger = logging.getLogger(__name__)


import gradio as gr


from __rag_pipeline import find_collections, find_documents

import __tech_about as tech_about
from __tech_chat import create_chat
from __tech_fn import sort_models, load_paths, load_tags, update_drop_down, update_system_prompt, update_textbox, update_chunks, import_setting, chatbot_avatars
from __tech_upload import create_upload


# Log all the variables as they get loaded
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
    user_tar = gr.State(value = IMAGES["user"])
    bot_tar = gr.State(value = IMAGES["bot"])

    # because of how many state variables I'm juggeling, I'm going to keep them alphabitized.
    with gr.Tabs():
        with gr.Tab(label = "About/Manual"):
            tech_about.about.render()
         
        with gr.Tab(label = "Technomancer Chat") as chat_tab:
            # TECH_CHAT, chat_components = create_chat(lang_models, rule_systems, system_content, embed_models)
            TECH_CHAT, chat_components = create_chat(lang_models, rule_systems, embed_models)


        with gr.Tab(label = "Database of Holding") as upload_tab:
            TECH_UPLOAD, upload_components = create_upload(db_paths, embed_models, rule_systems, tags)

    # upload_tab.select(fn = update_drop_down, inputs = [], outputs = [])
    upload_tab.select(fn = update_drop_down, inputs = [rule_systems], outputs = [upload_components["available_collections"]]).then(update_drop_down, [rule_systems], [upload_components["rule_system"]]).then(fn = update_drop_down, inputs = [embed_models], outputs = [upload_components["eb_model"]]).then(fn = update_drop_down, inputs = [db_paths], outputs = [upload_components["list_of_db"]])
    chat_tab.select(fn = update_drop_down, inputs = [lang_models], outputs = [chat_components["model_choice"]]).then(fn = update_drop_down, inputs = [rule_systems], outputs = [chat_components["collection_choice"]]).then(fn = update_drop_down, inputs = [embed_models], outputs = [chat_components["embedding_choice"]]).then(fn = chatbot_avatars, inputs = [user_tar, bot_tar], outputs = [chat_components["chatbot"]])



if __name__ in "__main__":
    Technomancer.launch(server_name = "0.0.0.0", server_port = 7860)