import gradio as gr
import logging
logger = logging.getLogger(__name__)


with gr.Blocks() as about:
    title = gr.Markdown("# About")
    chatbot = gr.Markdown("The chatbot provides a simple interactive way to find and reference the rules. Using a combination of Ollama (Phi4 for the model), LangChain (qwen3 for the embedding model), and ChromaDB, along with PDF Plumber, various rule books can be uploaded and stored for later reference. One challenge with this is the table nature of RPG books: they store a lot of important information in tables, and so adjusting both the chunk size and the chunk overlap is an important part of uploading any data.")
    upload = gr.Markdown("Uploading documents is fairly easy, however there is an art to properly ingesting a file. Due to how RPG rulebooks use so many tables, pictures, odd fonts, etc. it's not a one-size-fits-all for every document. By default the document is sliced every 4000 words, with a 200 word overlap, but this can be adjusted when the document on the upload page. It is important to audit the document after ingestion: make sure that the chatbot can properly retrieve the data. If it cannot, there is a button to delete documents.")
    copyright = gr.Markdown("There are a few games provided here by default. Those games are: Enchanted Realms and Super Hero Fun (both in Public Domain), and Codename: Spandex (which is under creative commons. Also this is a clone of Squadron UK, so if you like C:S check out that one.)")
    note_embedding = gr.Markdown("## Note about embedding model names")
    note_embedding_note = gr.Markdown("This will auto detect and sort your models by langauge and embedding models. This ASSUMES your embedding models have the word embedding in them: PLEASE DO NOT REMOVE THAT. Please leave embedding in the model name, which then allows you to easier select which model you want for the language portion and which for the embedding portion down the road.")
    # metadata = gr.Markdown(" For example, a group might be playing D&D 5e, but with home brew rules, and wants to pull up the grappling rules they wrote. Instead of getting the having to make several collections, sometimes with the same information, they can simple ask Technomancer about the grapple rules in their homebrew game, and Technomancer can search for both vanilla rules and homebrew ones.")

print("Rendered About page")

if __name__ in "__main__":
    about.launch()