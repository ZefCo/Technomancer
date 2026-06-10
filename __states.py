import logging
# from dotenv import load_dotenv
import gradio as gr
# import os
import pathlib
from __rag_pipeline import find_collections
from __tech_fn import import_settings, sort_models

logger = logging.getLogger(__name__)
logger.info(f"Reading States file @ (time to be implemented)")
cwd = pathlib.Path.cwd()

SETTINGS = import_settings()

RULE_SYSTEMS = find_collections()  # this will be the various collections in the database.
# DBOH = load_paths()
# TAGS = load_tags()
# IMAGES = import_setting(cwd / "Settings" / "AvatarImages.yaml")
IMAGES = tuple([SETTINGS["chatbot"]["user"], SETTINGS["chatbot"]["bot"]])
TAGS = SETTINGS["metadata"]["metadata"]
DBOH_SETTINGS = SETTINGS["database"]
DBOH = SETTINGS["database"]["default_path"]
DBOH_ALT = SETTINGS["database"]["alternative_paths"]

# DBOH_SETTINGS = import_setting(cwd / "Settings" / "DB_Settings.yaml")

# # LANG_MODELS, EMBED_MODELS = sort_models()

try:
    LANG_MODELS, EMBED_MODELS = sort_models(tuple(SETTINGS["embedd_models"]["models"]))
except Exception as e:
    print(f"Error type {type(e)} | Check if Ollama is installed and running | Creating dummy lists")
    LANG_MODELS = ["Dummy Language 1", "Dummy Language 2", "Dummy Language 3"]
    EMBED_MODELS = ["Dummy Embed 1", "Dummy Embed 2", "Dummy Embed 3"]

# Images Avatars
avatars_state: list = gr.State(value = IMAGES)  #[IMAGES["user"], IMAGES["bot"]])
# logger.info(f"Avatar state paths | {avatars_state}")

# Chunk sizes
chunk_batches_state: int = gr.State(value = DBOH_SETTINGS["chunk_batches_state"])
chunk_overlap_state: int = gr.State(value = DBOH_SETTINGS["chunk_overlap_state"])
chunk_size_state: int = gr.State(value = DBOH_SETTINGS["chunk_size_state"])
chunk_summary_state: int = gr.State(value = DBOH_SETTINGS["chunk_summary_state"])

# Paths to databases
db_paths_list_state: str = gr.State(value = DBOH)
# default message for typing
default_message_state: str = gr.State(value = "Type in a new rule system/collection")
# list of the documents for the selected rule system/collection
documents_list_state: list = gr.State(value = [])

# List of embedding models and the one seleted
embed_models_list_state: list = gr.State(value = EMBED_MODELS)
embed_model_state: str = gr.State(value = EMBED_MODELS[0])

# this is just to clear a drop down
empty_list_state: list = gr.State(value = [])

k_state: int = gr.State(value = DBOH_SETTINGS["k_state"])

lang_models_list_state: list = gr.State(value = LANG_MODELS)
lang_model_state: str = gr.State(value = LANG_MODELS[0])
log_file_state: str = gr.State(value = None)

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

percent_state: float = gr.State(value = DBOH_SETTINGS["percent_state"])
prefix_state: bool = gr.State(value = False)

rule_systems_list_state: list = gr.State(value = RULE_SYSTEMS)
rule_system_state: str = gr.State(value = RULE_SYSTEMS[0] if RULE_SYSTEMS else None)
# logger.info(f"Rule System State | {rule_system_state}")

settings_path_tags_state: str = gr.State(value = "Tags.yaml")

tags_list_state: list = gr.State(value = TAGS)
threshold_state: float = gr.State(value = DBOH_SETTINGS["threshold_state"])

upload_status_state: str = gr.State(value = "")

true_state: bool = gr.State(value = True)
false_state: bool = gr.State(value = False)

server_name = SETTINGS["server"]["server_name"]
server_port = SETTINGS["server"]["server_port"]