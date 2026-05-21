THIS IS NOT GENERATIVE AI!

Can it be used as such? I guess so. It's using an LLM to interact with several other features, and I guess you could add in 
something to generate RPG prompts, but that is not what this was designed for. This is meant to take all the tedious and 
overwhelming things in RPGs, like creating NPCs on the fly, and offload that to a computer program.

REQUIREMENTS

Ollama - uses Phi4, though other models can be used
    https://www.youtube.com/watch?v=GWB9ApTPTv4
    Will need a LLM model and an embedding model (example phi4 and qwen3-embedding:8B)
Gradio - for the UI
    https://www.youtube.com/watch?v=EON9jBnItUU
    https://medium.com/codex/building-your-first-ai-chatbot-with-ollama-and-gradio-e9667878941b
LangChain
    Also need langchain-ollama, langchain-community, pdfplumber, langchain-chroma, sentence-transformers, transformers (two different things)
    https://www.youtube.com/watch?v=1bUy-1hGZpI
    https://www.youtube.com/watch?v=tcqEUSNCn8I
ChromaDB
    https://www.youtube.com/watch?v=UuepzspChuQ
FastAPI (not implemented yet)
Some sort of relational database? SQL?
transformers

This will be broken up into a series of folders (which will be an ever growing list)

There will be a documentation folder for dealing with the various instructions, documentation, notes, etc. for how the overall project works.

Can this be pivoted into Rust?