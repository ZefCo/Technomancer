1) OVERVIEW

This is a locally hosted document retrieval and chat application, built with Ollama (for chat applications), LangChain (for RAG), ChromaDB (for vector embedding), and Gradio (for the web UI). Being that it is locally hosted, it can accessed on any device on the local network, and not meant to have any data leak out onto the web. It supports PDF ingestion, and the chunk size and overlap of the PDF can be adjusted on a per ingestion basis. Primarily it was designed as an assistant tool for referencing tables and rules in Role Playing Games, as they have a variety of tables and rules scattered throughout a large book with different fonts, making both in person referencing and the use of RAG a bit of an art form. This is also meant to be a "proof of concept" for larger and more technically challenging RAG pipelines that could be used, for example, in network maintenance, where the whole application is able to retrieve information on the network, process it, and respond with the state of different parts of the network. The overall application is meant as a tool for the user to make their work easier. Note that because this is a local LLM, it can be slow. Often the local LLM is going to take a few minutes to get started when it is first interacted with, and depending on the model you choose and your system abilities, it can taken even longer.

2) DISCLAIMER

Due to copyright laws, the database is not hosted here, and instead the user must build their own *with the books and documents that they have legally purchased.* Instructions are provided about how to do that, but again, they must bring their own legally purchased documents. RPG designers deserve to be in business, support them.

3) INSTALLATION & REQUIREMENTS

A list of requirements are in the requirements.txt file.

Ollama: can use any local model that the user has installed. Please note you have to bring your own Ollama model and it must be running in the background. This is meant for local LLMs, and has not been tested with internet connected ones. The choice of model you bring is going to have an impact on preformance. A small LLM will take up less space and have less hardware requirements than a bigger one, but will have a much slower response time. Feel free to bring different models as this application will search to see which ones are active, and sort them as embedding models and language models, allowing you to switch between them as you see fit. But the choice of model you bring is your responsibility.

Gradio: It runs as a local server on a device, which means any web capable device can interact with it at {that devices ip}:7860. It is not currently meant to go online as copyrighted material could be in it. For initial testing and deployment it will be populated with free RPG material, but even then the database (due to its size) will probably not be uploaded. Currently this does not support any login or chat saving, but both of those are planned for the future. At least locally. Please note! This requires Gradio 6 or later. 5 will note work! Gradio has made several critical changes that mean pervious versions of Gradio will be either expecting different message structures when interacting with the LLM or keyword arguements at runtime. Failure to run Gradio 6 will result in Technomancer crashing at start up (reporting a type error) or throwing an error when sending a message to the LLM.

LangChain: Also need langchain-ollama, langchain-community, pdfplumber, langchain-chroma, sentence-transformers, transformers (the last two are different things)

ChromaDB:
    
FastAPI (not implemented yet)

4) USE

4.1) MODELS

You can install any Ollama model you want, but you must install a language model and an embedding model. Multiple models can be installed, and you can switch between them. If you are familiar with how to make new models, you can also design your own. A model file title Modelfile_Technomancer is provided for creating a Technomancer personality for the chatbot, otherwise it will use whatever language model provided. No system content prompt is used, which allows for more flexibility with language model personalities. For embedding models, size does have an effect on preformance. Qwen 8b provides more nuance to retrival, but is slower, while Qwen 4b provides less nuance (and may come back with false positives) but is much faster. Overall, play with several different models until you find something you feel works well. The choice of language model matters greatly in the quality of responses you will recieve. Llama3.1:8b, for example, gives very matter of fact responses and tries to stick to the documents provided, while some others stray and begin to make up responses.

Note: there is a settings folder title EmbeddingModels.yaml that allows for the identification of embedding models (and by extension language models): this list is probably not comprehensive, and so if the embedding model you pick is not there, please add it. Additionally, while you can change your embedding model on the fly, be aware of what model you are using! Because models have different vertor representations, pulling information that one was embedded from one model will not translate well with another. It is recomended to have only one embedding model installed.

(The following is not yet implemented)
For advanced users, it is possible to user different databases and to switch embedding models for those different databases. However, only one embedding model maybe be used for a database (a database cannot suppoed multiple embedding models). When in doubt however, pick a single embedding model for all databases. 

4.2) WEB UI

To interact, make sure that the requirements are installed and satisfied (easily done in Anaconda), activate the python environment where those requirements are installed, and run start_up.py. This starts up the server on the local device at URL "0.0.0.0:7860". To get to the web interface, go to any web browser and use the IP of the host device at port 7860. For example, if the host device is at 123.456.7.890, the IP address for the web UI is 123.456.7.890:7860. If connecting *on* the host device, you can additionally use the local host ip with the port number, which is typically something like 127.0.0.1:7860. Currently there is no login, you are brought straight to the splash page. 

Right now the chat is not persistent, and will be refreshed once the page is refreshed. The ability to save chat is forthcoming. Also that a system prompt can be adjusted in the Advanced Features section of the chatbot page, however doing so *mid conversation* can cause issues with the chatbot. It can become confused/hallucinate because it was in the middle of one conversation and now told to do something different, but is still able to see previous portions of the chat. While this can be a useful feature, it is advised to use it sparingly. Currently the only model supported is phi4, so check system requirements on that before downloading. Eventually other chatbots will be available for selection.

4.3) DOCUMENT UPLOAD & RETRIEVAL

This is mostly designed as a document retrieval program to quickly reference and pull existing rules, notes, tables, etc. from a database. This is not meant as generative AI: I am not trying to replace imagination here, just make getting rules easier.

Can it be used as such? I guess so. It's using an LLM to interact with several other features, so it would be possible to add in something to generate RPG prompts, but that is not what this was designed for. This is meant to take all the tedious and overwhelming things in RPGs, like creating NPCs on the fly, and offload that to a computer program. It is meant as a tool to make the workload easier.

This is meant to quickly retrieve and interact with various documents, allowing easy access to rules and tables in an RPG book. RPG books have a variable amount of rules, some have lots (crunchy), some don't (rules-lite). Also they have lots of tables, which store rules, modifiers, random generators, etc. Instead of just having to use post it notes, an index, or just waste a lot of time thumbing through to find what you're looking for, this provides a very quick and easy way to grab the specific rule or table you are looking for.

Loading documents is fairly easy: got to the Database of Holding, and drag and drop a file to the upload space. Depending on file size, the upload will take some time to complete. The documents are chunked, which can be adjusted via a slider (along with the chunk overlap), and will be uploaded in chunks (which can also be adjusted via a slider). The time it takes up upload a document is based on computer preformance, and due to the size of many pdfs, this can take awhile. The model also effects this heavily. Note: increasing batch size too much can actually slow things down even further. 

Can you upload multiple documents at once? No. Will that be implemented in the future? Also no. Should it be? Again, due ot the size of the documents, uploading can take a long while, so mass uploading is not of interest here.

By default the documents are uploaded in chunks of 256 words with 50 overlap. This can be adjusted in the "Upload" page, but does not effect the chatbot upload document option. This ability to adjust the chunks is because, again, due to the structure of RPG rule books, the chunk size cannot be a one size fits all number. By default the chunk size is 256 and the chunk overlap is 50. It is important to audit what was uploaded by asking Technomancer questions about what was uploaded and making sure the response is the intended answer. If not, delete the document and try again with a different  chunk size (Right now deleting specific chunks is not possible).

A good rule of thumb for the chunk size and overlap:

128 - 256 is good for fact based queries where prevision is advised.
256-512 is a good general purpose range
512 - 1024 is good for technical documents like research papers.

For the chunk overlap, 10-20% of the chunk size is great.

5) NOTE ON COLLECTIONS

Normally there are rules about how the named collections can be formated. To deal with this, all named collections are turned into an ascii string, which should preserve the original rule book system name. For this reason, the collection name for rule books should be limited to a maximum length of 120 characters. Normally it is allowed to be 3 to 512 characters for a ChromaDB collection, however, since they are being turned into an ascii representation of 2-3 numbers which are then spaced by an underscore, this really limits it to 512/4 = 128. In practice it should be limited to 120, just in case an ascii representation is greater than 4.

As of 5/28/26, to use the RAG pipeline, the collection must be selected directly. This allows the user to select either a specific collection or general chat. The downside is that this requires more manual input then desired.
