import logging
import gradio as gr
import pathlib
from __rag_pipeline import find_collections
from __tech_fn import import_setting, load_paths, load_tags, sort_models
logger = logging.getLogger(__name__)
logger.info(f"Reading States file @ (time to be implemented)")

cwd = pathlib.Path.cwd()

RULE_SYSTEMS = find_collections()  # this will be the various collections in the database.
LANG_MODELS, EMBED_MODELS = sort_models()
DBOH = load_paths()
TAGS = load_tags()
IMAGES = import_setting(cwd / "Settings" / "AvatarImages.yaml")

# Images Avatars
avatars_state = gr.State(value = [IMAGES["user"], IMAGES["bot"]])
# logger.info(f"Avatar state paths | {avatars_state}")

# Chunk sizes
chunk_batches_state = gr.State(value = 50)
chunk_overlap_state = gr.State(value = 50)
chunk_size_state = gr.State(value = 512)

# Paths to databases
db_paths_list_state = gr.State(value = DBOH)
# default message for typing
default_message_state = gr.State(value = "Type in a new rule system/collection")
# list of the documents for the selected rule system/collection
documents_list_state = gr.State(value = [])

# List of embedding models and the one seleted
embed_models_list_state = gr.State(value = EMBED_MODELS)
embed_model_state = gr.State(value = EMBED_MODELS[0])

# this is just to clear a drop down
empty_list_state = gr.State(value = [])


lang_models_list_state = gr.State(value = LANG_MODELS)
lang_model_state = gr.State(value = LANG_MODELS[0])

name_chunksize_state = gr.State(value = "Chunk Size")
named_chunkoverlap_state = gr.State(value = "Chunk Overlap")
name_embed_state = gr.State(value = "Embedding Model")
name_lang_state = gr.State(value = "Language Model")
name_rule_state = gr.State(value = "Rule System")
name_tags_state: str = gr.State(value = "Metadata Tags")

overlap_name_state = gr.State(value = "Chunk Overlap")

percent_state = gr.State(value = 0.1)

rule_systems_list_state = gr.State(value = RULE_SYSTEMS)
rule_system_state = gr.State(value = RULE_SYSTEMS[0] if RULE_SYSTEMS else None)
# logger.info(f"Rule System State | {rule_system_state}")

settings_path_tags_state: str = gr.State(value = "Tags.yaml")

tags_list_state: list = gr.State(value = TAGS)

upload_status_state = gr.State(value = "")

true_state = gr.State(value = True)
false_state = gr.State(value = False)