import logging
# from dotenv import load_dotenv
import gradio as gr
# import os
import pathlib
from __rag_pipeline import find_collections
from __tech_fn import import_settings, sort_models, user_logins

logger = logging.getLogger(__name__)
logger.info(f"Reading States file @ (time to be implemented)")
cwd = pathlib.Path.cwd()

SETTINGS = import_settings()

ALL_USERS = user_logins()

try:
    RULE_SYSTEMS = find_collections()  # this will be the various collections in the database.
except Exception as e:
    import sys
    print(f"Unable to find rule system - Check if Chroma is Running.\nType error = {type(e)}")
    if sys.argv[1] == "DEBUG":
        logger.critical(f"Entering DEBUG mode | Generating dummy rules")
        RULE_SYSTEMS = ["Dummy Rules 1", "Dummy Rule 2", "Dummy Rules 3"]
    else:
        raise
# DBOH = load_paths()
# TAGS = load_tags()
# IMAGES = import_setting(cwd / "Settings" / "AvatarImages.yaml")
IMAGES = tuple([SETTINGS["chatbot"]["user"], SETTINGS["chatbot"]["bot"]])
TAGS = SETTINGS["metadata"]["metadata"]
DBOH_SETTINGS = SETTINGS["database"]

# DBOH_SETTINGS = import_setting(cwd / "Settings" / "DB_Settings.yaml")

# # LANG_MODELS, EMBED_MODELS = sort_models()

try:
    LANG_MODELS, EMBED_MODELS, VISION_MODELS = sort_models(tuple(SETTINGS["embedd_models"]["models"]), tuple(SETTINGS["multimodal"]["models"]), SETTINGS["ports"]["ollama"])
except Exception as e:
    import sys
    print(f"Error type {type(e)} | Check if Ollama is installed and running")
    if sys.argv[1] == "DEBUG":
        logger.critical(f"Error type {type(e)} | Check if Ollama is installed and running | Creating dummy lists")
        LANG_MODELS = ["Dummy Language 1", "Dummy Language 2", "Dummy Language 3"]
        EMBED_MODELS = ["Dummy Embed 1", "Dummy Embed 2", "Dummy Embed 3"]
        VISION_MODELS = ["Dummy Vision 1", "Dummy Vision 2", "Dummy Vision 3"]
    else:
        raise
else:
    if len(LANG_MODELS) < 1: LANG_MODELS = ["Dummy Language 1", "Dummy Language 2", "Dummy Language 3"]
    if len(EMBED_MODELS) < 1: EMBED_MODELS = ["Dummy Embed 1", "Dummy Embed 2", "Dummy Embed 3"]
    if len(VISION_MODELS) < 1: VISION_MODELS = ["Dummy Vision 1", "Dummy Vision 2", "Dummy Vision 3"]

# <-- Global States -->


# List of embedding models and the one seleted
embed_models_list_state: list = gr.State(value = EMBED_MODELS)

# this is just to clear a drop down
empty_list_state: list = gr.State(value = [])

lang_models_list_state: list = gr.State(value = LANG_MODELS)

rule_systems_list_state: list = gr.State(value = RULE_SYSTEMS)

tags_list_state: list = gr.State(value = TAGS)

vision_models_list_state: list = gr.State(value = VISION_MODELS)

# <-- these ones don't really matter -->
name_chunkbatch_state: str = gr.State(value = "Chunk Batch")
name_chunksize_state: str = gr.State(value = "Chunk Size")
named_chunkoverlap_state: str = gr.State(value = "Chunk Overlap")
name_chunksum_state: str = gr.State(value = "Chunk Summary")
name_embed_state: str = gr.State(value = "Embedding Model")
name_k_state: str = gr.State(value = "K")
name_lang_state: str = gr.State(value = "Language Model")
name_rule_state: str = gr.State(value = "Rule System")
name_tags_state: str = gr.State(value = "Metadata Tags")
name_threshold_state: str = gr.State(value = "Cosine Similarity Threshold")
name_vis_state: str = gr.State(value = "Vision Model")
true_state: bool = gr.State(value = True)
false_state: bool = gr.State(value = False)

# <-- User States -->
# Images Avatars
avatars_state: list = gr.State(value = IMAGES)  #[IMAGES["user"], IMAGES["bot"]])

# Chunk sizes
chunk_batches_state: int = gr.State(value = DBOH_SETTINGS["chunk_batches_state"])
chunk_overlap_state: int = gr.State(value = DBOH_SETTINGS["chunk_overlap_state"])
chunk_size_state: int = gr.State(value = DBOH_SETTINGS["chunk_size_state"])
chunk_summary_state: int = gr.State(value = DBOH_SETTINGS["chunk_summary_state"])

# default message for typing
default_message_state: str = gr.State(value = "Type in a new rule system/collection")

# list of the documents for the selected rule system/collection
documents_list_state: list = gr.State(value = [])

# For the currently selected embedding model
embed_model_state: str = gr.State(value = EMBED_MODELS[0])

# Default k value
k_state: int = gr.State(value = DBOH_SETTINGS["k_state"])

# What language model is being used
lang_model_state: str = gr.State(value = LANG_MODELS[0])

# What percent is being used and if the prefix is being used. As the prefix has given me trouble, it's hard coded to False right now.
percent_state: float = gr.State(value = DBOH_SETTINGS["percent_state"])
prefix_state: bool = gr.State(value = False)

# Which rule system is being used.
rule_system_state: str = gr.State(value = RULE_SYSTEMS[0] if RULE_SYSTEMS else None)
# logger.info(f"Rule System State | {rule_system_state}")

# Are the chunks and the summary being saved.
save_chunk_state: bool = gr.State(value = True)
save_sum_state: bool = gr.State(value = True)

# The threshold for responses.
threshold_state: float = gr.State(value = DBOH_SETTINGS["threshold_state"])

# What is the current upload status.
upload_status_state: str = gr.State(value = "")

vision_model_state: str = gr.State(value = VISION_MODELS[0])

