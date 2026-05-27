REQUIREMENTS

Ollama - uses Phi4, though other models can be used
    currently uses phi4 and qwen3 for language and embedding. This is hard coded, though there are plans to make this more adaptable.
Gradio - for the UI
    It runs as a local server on a device, which means any web capable device can interact with it at {that devices ip}:7860.
    It is not currently meant to go online as copyrighted material could be in it. For initial testing and deployment it will be
    populated with free RPG material, but even then the database (due to its size) will probably not be uploaded.
    Currently this does not support any login or chat saving, but both of those are planned for the future.
LangChain
    Also need langchain-ollama, langchain-community, pdfplumber, langchain-chroma, sentence-transformers, transformers (two different things)
    This allows Ollama and ChromaDB to interact.
ChromaDB
    For holding the database.
FastAPI (not implemented yet)
transformers

This is not generative AI.

Can it be used as such? I guess so. It's using an LLM to interact with several other features, and I guess you could add in 
something to generate RPG prompts, but that is not what this was designed for. This is meant to take all the tedious and 
overwhelming things in RPGs, like creating NPCs on the fly, and offload that to a computer program. But that is not what it was
meant to be.

What it can do: this is meant to quickly retrieve and interact with various documents, allowing easy access to rules and tables
in an RPG book. RPG books have a variable amount of rules, some have lots (crunchy), some don't (rules-lite). Also they have lots
of tables, which store rules, modifiers, random generators, etc. Instead of just having to use post it notes, an index, or just
waste a lot of time thumbing through to find what you're looking for, this provides a very quick and easy way to grab the specific
rule or table you are looking for.

Documents can be uploaded at will, and the chunk size and overlap can be adjusted: this is because, again, due to the structure of
RPG rule books, the chunk size cannot be a one size fits all number. By default the chunk size is 4000 and the chunk overlap is 200,
but this can be adjusted with sliders in the upload page. It is important to audit what was uploaded by asking Technomancer 
questions about what was uploaded and making sure the reponse is the inteded answer. If not, delete the document and try again with
a different chunk size.