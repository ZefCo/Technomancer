import gradio as gr
print("Finished loading About Page")

with gr.Blocks() as page:
    title = gr.HTML("<h1>About</h1>")
    chatbot = gr.HTML("<section>" \
                      "<h2>The Chatbot</h2>"
                      "<p>The chatbot provides a simple interactive way to find and reference the rules. Using a combination of Ollama (Phi4 for the model), LangChain (qwen3 for the embedding model), and ChromaDB, along with PDF Plumber, various rule books can be uploaded and stored for later reference. One challenge with this is the table nature of RPG books: they store a lot of important information in tables, and so adjusting both the chunk size and the chunk overlap is an important part of uploading any data.</p>")
    upload = gr.HTML("<section>"
                     "<h2> Uploading Documents </h2>"
                     "<p>Uploading documents is fairly easy, however there is an art to properly ingesting a file. Due to how RPG rulebooks use so many tables, pictures, odd fonts, etc. it's not a one-size-fits-all for every document. By default the document is sliced every 4000 words, with a 200 word overlap, but this can be adjusted when the document on the upload page. It is important to audit the document after ingestion: make sure that the chatbot can properly retrieve the data. If it cannot, there is a button to delete documents.")
    other = gr.HTML("<section>"
                    "<h2> Other Features </h2>"
                    "<p>As of 5/31/26, only the chatbot and uploading of documents are working. Eventually other features will be added, such as an NPC generator for certain games.")
    copyright = gr.HTML("<section>"
                        "<h2> Hosted Games </h2>"
                        "<p> There are a few games provided here by default. Those games are: Enchanted Realms and Super Hero Fun (both in Public Domain), and Codename: Spandex (which is under creative commons. Also this is a clone of Squadron UK, so if you like C:S check out that one.)")

if __name__ in "__main__":
    page.launch()