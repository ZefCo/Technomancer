# States

Getting a list of all the states and where they are used

## Global

The available embedding models in the system. This is found once at the start and passed to the Chat Tab and Upload Tab
* embed_models_list_state: list = gr.State(value = EMBED_MODELS)

Just like the embedding models, except for language models. Both these could actually be in the Other since they are static. I'll put these on the back burner right now.
* lang_models_list_state: list = gr.State(value = LANG_MODELS)

The rule systems. This should be pulled from the collections, and will be updated if something is added or removed.
* rule_systems_list_state: list = gr.State(value = RULE_SYSTEMS)

Similar to the rule systems, but this is the metadata tags.
* tags_list_state: list = gr.State(value = TAGS)


## User
These are things that are user dependent. The users can adjust these and change them from one another.

This is list of documents currently available to the user in the selected collection.
* documents_list_state: list = gr.State(value = [])

Images Avatars
* avatars_state: list = gr.State(value = IMAGES)

Chunk sizes
* chunk_batches_state: int = gr.State(value = DBOH_SETTINGS["chunk_batches_state"])
* chunk_overlap_state: int = gr.State(value = DBOH_SETTINGS["chunk_overlap_state"])
* chunk_size_state: int = gr.State(value = DBOH_SETTINGS["chunk_size_state"])
* chunk_summary_state: int = gr.State(value = DBOH_SETTINGS["chunk_summary_state"])

default message for typing
* default_message_state: str = gr.State(value = "Type in a new rule system/collection")

For the currently selected embedding model
* embed_model_state: str = gr.State(value = EMBED_MODELS[0])

Default k value
* k_state: int = gr.State(value = DBOH_SETTINGS["k_state"])

What language model is being used
* lang_model_state: str = gr.State(value = LANG_MODELS[0])

What percent is being used and if the prefix is being used. As the prefix has given me trouble, it's hard coded to False right now.
* percent_state: float = gr.State(value = DBOH_SETTINGS["percent_state"])
* prefix_state: bool = gr.State(value = False)

Which rule system is being used.
* rule_system_state: str = gr.State(value = RULE_SYSTEMS[0] if RULE_SYSTEMS else None)

Are the chunks and the summary being saved.
* save_chunk_state: bool = gr.State(value = True)
* save_sum_state: bool = gr.State(value = True)

The threshold for responses.
* threshold_state: float = gr.State(value = DBOH_SETTINGS["threshold_state"])

What is the current upload status.
* upload_status_state: str = gr.State(value = "")


## Other
These can go either way. They are global states, but it's easier to handle them as static user states.

* name_chunkbatch_state: str = gr.State(value = "Chunk Batch")
* name_chunksize_state: str = gr.State(value = "Chunk Size")
* named_chunkoverlap_state: str = gr.State(value = "Chunk Overlap")
* name_chunksum_state: str = gr.State(value = "Chunk Summary")
* name_embed_state: str = gr.State(value = "Embedding Model")
* name_k_state: str = gr.State(value = "K")
* name_lang_state: str = gr.State(value = "Language Model")
* name_rule_state: str = gr.State(value = "Rule System")
* name_tags_state: str = gr.State(value = "Metadata Tags")
* name_threshold_state: str = gr.State(value = "Cosine Similarity Threshold")
* true_state: bool = gr.State(value = True)
* false_state: bool = gr.State(value = False)
* empty_list_state: list = gr.State(value = [])
    * This is just to clear a drop down. In fact it might not even be used. I'll keep this here though for later since it could be useful.
