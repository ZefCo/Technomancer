from datetime import datetime
import pathlib
import logging
logger = logging.getLogger(__name__)
logger.info(f"Reading RAG Pipeline file @ (time to be implemented)")


import chromadb


from functools import lru_cache

from langchain_chroma import Chroma

from langchain_community.document_loaders import PDFPlumberLoader, TextLoader, UnstructuredCSVLoader, UnstructuredEPubLoader, UnstructuredWordDocumentLoader, UnstructuredMarkdownLoader

from langchain_ollama import OllamaEmbeddings  # this should hopefully get rid of that warning about depreciation
from langchain_ollama import ChatOllama

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableBranch, RunnableLambda, RunnablePassthrough

from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter

import ollama

cwd = pathlib.Path.cwd()
chroma_database_dir = cwd / "DB_of_Holding"

_chroma_client = None


def _classify_and_tag(msg: str, llm, rules_systems: list[str], available_tags: list[str]):
    '''
    Classifies the message and adds the metadata tags to the LMM.

    Tweek this to handle the possibility of different editions. Let the LLM output a list of rules and it can choose the most probable one later.
    '''
    prompt = ChatPromptTemplate.from_messages([
        ("system", f"""Analyze the user query and respond with exactly three lines and nothing else. Do not add preamples, extra lines, or anything else.
         Line 1: Either RULES or GENERAL
         Line 2: Comma-separated relevant tags from this list: {", ".join(available_tags)}, or None
         Line 3: the name of the rule system use from this list: {rules_systems}, or NONE

         RULES means questions about game mechanics, rules, tables, states, abilities, NPCs, lore, or anything from a rulebook.
         GENERAL means everything else.

         Example response:
         RULES
         Rules,Combat
         D&D
         """), ("human", "{question}")
    ])

    result = (prompt | llm | StrOutputParser()).invoke({"question": msg}).strip()
    lines = result.strip().splitlines()

    classification = lines[0].strip().upper() if lines else "GENERAL"
    
    tags = []
    if len(lines) > 1 and lines[1].strip().upper() != "NONE":
        tags = [t.strip() for t in lines[1].split(",") if t.strip() in available_tags]

    game_system = None
    if len(lines) > 2 and lines[2].strip().upper() != "NONE":
        game_system = lines[2].strip()
        game_system = game_system if game_system in rules_systems else None

    logger.info(f"Classified Question | {classification} | {tags} | {game_system}")

    return classification, tags, game_system


def _clean_collection(collection: str):
    '''
    Turns the collection into a valid name, which will be ascii values. Each ascii value is separated by an _
    
    Because it turns it into a series of ascii values in the form x_x_x_x where each x is an ascii value of 2-3 numbers, this means the input string must be less than 128 characters long. That's OK. I doubt anyone will want to title a collection that long.
    '''
    # ascii_collection = f"{ord(collection[0])}"
    # for c in collection[1:len(collection)]:
    #     ascii_collection = f"{ascii_collection}_{ord(c)}"

    logger.info(f"Input Collection String name | {collection}")
    ascii_collection = "_".join(str(ord(c)) for c in collection)
    logger.info(f"Output Collection ASCII name | {ascii_collection}")


    if len(ascii_collection) < 3:
        for _ in range(3 - len(ascii_collection)): ascii_collection = ascii_collection + f"_{ord("z")}"
        logger.warning(f"Original collection string was too short, extending by z = {ord("z")}")


    if len(ascii_collection) > 512:
        ascii_collection = ascii_collection[:512]
        logger.warning(f"Original collection was too long (in ascii), truncating.")

    return ascii_collection


def _create_chunks(document, chunk_size, chunk_overlap, *args, **kwargs):
    '''
    Chunks the document

    Part of RAG Input
    '''
    logger.info(f"Ingesting splitting file into chunks | Chunk Size: {chunk_size} | Chunk Overlap: {chunk_overlap}")
    split_text = RecursiveCharacterTextSplitter(chunk_size = chunk_size, chunk_overlap = chunk_overlap)

    chunks = split_text.split_documents(document)

    if not chunks:
        logger.warning(f"No chunks were created from the document")
        return None
    else:
        logger.info(f"{len(chunks)} were created from document")
    
    return chunks


# Add to log that this is happeneing
def create_collection(collection: str):
    '''
    Creates an empty collection if that collection name is not already in use.
    '''
    logger.info(f"Creating new Collection in Database")
    collection = _clean_collection(collection)
    client = _get_client()
    try:
        _ = client.get_or_create_collection(name = collection)
    except Exception as e:
        logger.critical(f"Error creating collection {collection} in database")
    else:
        logger.info(f"Successfully created collection in database")

    
def delete_collection(hr_collection, nuclear_option: bool = False):
    '''
    Deletes an entire collection of documents.
    '''
    if hr_collection is None:
        return None

    client = _get_client()

    if nuclear_option:
        logger.critical(f"Everything is being reset!")
        client.reset()
        logger.critical(f"Everything in the database was removed | Database is not reset")
        return None

    ascii_collection = _clean_collection(hr_collection)
    logger.warning(f"Entire collection being deleted! | {hr_collection} | {ascii_collection}")

    client.delete_collection(name = ascii_collection)
    logger.warning(f"{hr_collection} was deleted | {ascii_collection} was removed")



def delete_document(hr_collection, metadata):
    '''
    Deletes a document in the database. Does so by finding everything with matching metadata and deleting it. Searches on the source of the data, which is hopefully reliable.

    Again, use the human readable version of the collection name.

    Metadata is really the title. The variable is named as such because it's pulling from the metadata toget the title.
    '''
    client = _get_client()
    
    ascii_collection = _clean_collection(hr_collection)
    logger.info(f"Deleting {metadata} | {hr_collection} | {ascii_collection}")
    local_ascii_collection = client.get_collection(str(ascii_collection))
    
    try:
        local_ascii_collection.delete(where = {"Title": metadata})
    except Exception as e:
        logger.critical(f"Error deleting chunks for the document with metadata {metadata} | {hr_collection} | {type(e)} | {e}")
    else:
        documents = find_documents(hr_collection)
        logger.info(f"Documents left | {hr_collection} | {documents}")


def _direct_response(message: str, history: list, lang_model: str):
    '''
    Fallback for General questions, in case the question is classified as GENERAL. This is pretty similar to the other response in __tech_fn.py
    '''
    messages = []
    for item in _gradio_history_to_langchain(history):
        if isinstance(item, HumanMessage):
            messages.append({"role": "user", "content": item.content})
        elif isinstance(item, AIMessage):
            messages.append({"role": "assistant", "content": item.content})
    
    messages.append({"role": "user", "content": message})

    completion = ollama.chat(model = lang_model, messages = messages, stream = True)
    response = ""

    for chunk in completion:
        if "message" in chunk and "content" in chunk["message"]:
            response += chunk["message"]["content"]
            yield response


def _enrich_chunk_metadata(chunks, game_system: str, llm_model: str | None = None):
    '''
    Tags specific documents with certain tags, making sections easier to identify later.
    '''
    from __tech_fn import import_setting
    keywords: dict = import_setting("Keyword.yaml")
    for chunk in chunks:
        text = chunk.page_content.lower()
        auto_tags = set()

        if _looks_like_table(chunk.page_content):
            chunk.metadata["chunk_type"] = "table"
            auto_tags.add("table")
        elif _looks_like_stat_block(chunk.page_content):
            chunk.metadata["chunk_type"] = "stat_block"
            auto_tags.add("stat_block")
        else:
            chunk.metadata["chunk_type"] = "text"

        for term, tag_terms in keywords.items():
            if set(tag_terms) & set(text.split()):
                auto_tags.add(term)

        chunk.metadata["game_system"] = game_system
        chunk.metadata["auto_tags"] = ",".join(sorted(auto_tags))

        existing_tags = set(chunk.metadata.get("tags", "").split(",")) - {""}
        chunk.metadata["tags"] = ",".join(sorted(existing_tags | auto_tags))

    return chunks


def find_collections():
    '''
    Returns all collections in the database.

    Checks to see if anything is present. If there is, it will clean up the collection name into a human readable format and return it. If there is nothing, just returns an empty list.
    '''
    client = _get_client()
    collections = client.list_collections()
    if collections: 
        collections = [human_collection(collection.name) for collection in collections]
        logger.info(f"Collections found in database: {collections}")
    else:
        logger.info(f"No collections found in database")

    return collections


# Log if the collections found are empyt or not. Send lenth of list of titles to log.
def find_documents(hr_collection):
    '''
    Finds all available documents in a given collection. Feed in the human readable collection title.
    '''
    titles = set()
    ascii_collection = _clean_collection(hr_collection)

    client = _get_client()
    local_collection = client.get_collection(str(ascii_collection))
    results: dict = local_collection.get() # type: ignore  There's an error here that shouldn't be an error. It works fine.

    # print(list(results.keys()))
    # first = list(results["metadatas"][0])
    # first.sort()
    # print(first.get("Title", None))
    # ['ids', 'embeddings', 'documents', 'uris', 'included', 'data', 'metadatas']
    # From metadatas: ['Author', 'CreationDate', 'Creator', 'ModDate', 'Producer', 'Title', 'file_path', 'game_system', 'id', 'page', 'source', 'tags', 'total_pages']

    items = 0
    for item in results["metadatas"]:
        title = item.get("Title", None)  # This shouldn't be a thing anymore: I'm manually adding titles if the title isn't there. But just in case I do miss something, I'm going to keep this code written as is.
        try:
            if title: titles.add(title)
            else: titles.add(pathlib.Path(item["source"]).stem)  # this is in case the title doesn't have a Title.
        except Exception as e:
            logger.warning(f"Cannont find title or source in item retreived from collection {hr_collection} | Error type {type(e)} | {e}")
        else:
            items += 1
    logger.info(f"Successfully found {items} | collection {ascii_collection}| {hr_collection}")
    
    return list(titles)


def _format_docs(docs):
    '''
    Formats retrieved documents into a single context string. Includes source metadata so the LLM can cite pages.

    Part of RAG Query
    '''
    formatted = []
    for doc in docs:
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "?")
        formatted.append(f"[Source: {source}, Page: {page}]\n{doc.page_content}")
    return "\n\n".join(formatted)


def _generate_section_summary(chunks: list, lang_model: str, section_size: int = 10) -> list:
    '''
    Generates a summary from an LLM about the chunks.
    '''
    logger.info(f"Generating Summary | {lang_model} | Section Size: {section_size}")
    summary_docs = []
    section_index = -1
    game_system = chunks[0].metadata["game_system"]
    file_path = chunks[0].metadata["source"]
    base_id = f"{str(pathlib.Path(file_path).stem)}:summary"


    for i in range(0, len(chunks), section_size):
        section_index += 1
        section = chunks[i: i + section_size]
        combined_text = ""
        combined_tags = set()
        for c in section:
            # combined_text = "\n\n".join(c.page_content for c in section)
            combined_text = f"{combined_text}\n\n{c.page_content}"
            combined_tags = combined_tags | set(c.metadata["tags"].split(","))

        response = ollama.chat(model = lang_model,
                               messages = [{"role": "user",
                                            "content": f"""Summarize the following rulebook sections in 2-5 paragraphs. Generate tables or lists if necessary.
                                            Forcus on: what rules or mechanics are covered, what a player needs to know, any key terms that should be defined. Be concise but complete.

                                            Text:
                                            {combined_text}"""}])
        summary_text = response["message"]["content"]

        first_chunk = section[0]
        
        summary_doc = Document(page_content = summary_text,
                               metadata = {
                                           "game_system": game_system,
                                           "tags": ",".join(sorted(combined_tags)),
                                           "chunk_type": "summary",
                                           "source": file_path,
                                           "source_pages": f"{section[0].metadata.get('page', '?')}-{section[-1].metadata.get('page', '?')}",
                                           "original_chunk_count": len(section),
                                           "id": f"{base_id}:{section_index}"
                                           })
        
        summary_docs.append(summary_doc)
        logger.info(f"Generated summary | {first_chunk.metadata.get('page', '?')} - {section[-1].metadata.get('page', '?')}")

    return summary_docs


# Log when this is called and creates a new client.
def _get_client():
    '''
    Connects to the client
    '''
    # https://stackoverflow.com/questions/77134962/connecttimeout-error-when-connecting-to-a-chromadb-client-that-is-hosted-on-azur
    # Add connection issues to the client. Send those to the logs
    logger.info(f"Connecting to client")
    global _chroma_client
    if _chroma_client is None: _chroma_client = chromadb.PersistentClient(path = str(chroma_database_dir))
    return _chroma_client


@lru_cache(maxsize=4)
def _get_embeddings(embed_model, search_doc = True):
    '''
    Currently this is hard coded to only use the qwen3 model embeddings.

    Part of RAG Input
    '''
    if search_doc:
        return OllamaEmbeddings(model = embed_model, model_kwargs = {"prompt": "search_document:"})
    return OllamaEmbeddings(model = embed_model)


# def _get_multi_retriever(collections: list[str], k: int = 5):
#     '''
#     Returns a single retriever that searches across multiple collections. Mereges results and returns the top k across all of them.
#     '''
#     from langchain_core.retrievers import MergerRetriever

#     retrievers = [_get_retriever(col, k = k) for col in collections]

#     return MergerRetriever(retrievers = retrievers)


def get_metadata(hr_collection: str, title: str):
    '''
    Gets the metadata for a given document.
    '''
    ascii_collection = _clean_collection(hr_collection)
    client = _get_client()
    collection = client.get_collection(name = ascii_collection)

    logger.info(f"Getting metadata tags for {title} | {hr_collection} | {collection}")

    results = collection.get(where = {"Title": title})
    try:
        local_tags = results["metadatas"][0]["tags"]  # if this is done right, the only thing that needs to be pulled is the first index. They all should have the same metadatas
    except IndexError as e:
        logger.error(f"Index Error trying to pull metadata | {title} | {hr_collection} | {type(results)} | {len(results)}")
        return []
    except Exception as e:
        logger.error(f"Error trying to pull metadatas | {title} | {hr_collection} | {ascii_collection} | {type(results)} - expected to be type dict | {type(e)} | {e}")
        return []

    logger.info(f"Returning tags | {local_tags}")

    return local_tags.split(",")


# def get_title(entry):
#     '''
#     Returns a single entry in the database.
#     '''
#     title = item.get("Title", None)
#     for item in results["metadatas"]:
#         title = item.get("Title", None)
#         try:
#             if title: titles.add(title)
#             else: titles.add(pathlib.Path(item["source"]).stem)  # this is in case the title doesn't have a Title.
#         except Exception as e:
#             logger.warning(f"Cannont find title or source in item retreived from collection {hr_collection} | Error type {type(e)} | {e}")
#         else:


@lru_cache(maxsize = 8)
def _get_retriever(hr_collection: str, embed_model: str, 
                   k: int = 10, tags: tuple[str, ...] | None = None, score_threshold: float | None = 0.3,  # score is 1 - cosine similarity, so a lower number here is a higher threshold
                   search_query: bool = True, *args, **kwargs):
    '''
    Returns a retriever for the given collection name.

    Part of RAG Query
    '''
    # if isinstance(tags, tuple): tags = list(tags)
    if not hr_collection: hr_collection = "Generic"
    ascii_collection = _clean_collection(hr_collection)
    QEM = _get_query_embeddings(embed_model, search_query)
    db = Chroma(persist_directory = str(chroma_database_dir), embedding_function = QEM, collection_name = ascii_collection)

    search_kwargs = {"k": k}

    filters = [{"game_system": hr_collection}]

    if tags:
        tag_filters = [{"tags": {"$contains": tag}} for tag in tags]
        if len(tag_filters) == 1:
            filters.append(tag_filters[0])
        else:
            filters.append({"$or": tag_filters})

    if len(filters) == 1:
        search_kwargs["filter"] = filters[0]
    else:
        search_kwargs["filter"] = {"$and": filters}
    
    logger.info(f"k = {k} | score threshold = {score_threshold} (1 - {score_threshold} is used)")

    if score_threshold is not None:
        return db.as_retriever(search_type = "similarity_score_threshold",
                               search_kwargs = {**search_kwargs, "score_threshold": score_threshold})
    
    return db.as_retriever(search_type = "similarity", search_kwargs = search_kwargs)
    
    # return db.as_retriever(search_type = "similarity", search_kwargs = {"k": k, "filter": {"game_system": hr_collection}})


def _get_summary_retriever(hr_collection: str, embed_model: str, 
                           k: int = 5, tags: list[str] | None = None, score_threshold: float | None = 0.3):
    '''
    Limits its retriever to only the summary chunks.
    '''
    ascii_collection = _clean_collection(hr_collection)

    db = Chroma(persist_directory=str(chroma_database_dir), embedding_function=_get_query_embeddings(embed_model), collection_name=ascii_collection)

    filters = [{"game_system": hr_collection}, {"chunk_type": "summary"}]

    if tags:
        tag_filters = [{"tags": {"$contains": tag}} for tag in tags]
        filters.append({"$or": tag_filters} if len(tag_filters) > 1 else tag_filters[0])
    
    search_kwargs = {"k": k, "filter": {"$and": filters}}

    if score_threshold is not None:
        return db.as_retriever(search_type = "similarity_score_threshold", search_kwargs = {**search_kwargs, "score_threshold": score_threshold})

    return db.as_retriever(search_type = "similarity", search_kwargs = search_kwargs)

    
def _get_query_embeddings(embeddings: str, search_query: bool = True):
    '''
    This allows for the use of the search query prefix in nomic-embed-text. The others, mixed bread and snowflake artic, will come eventually.
    '''
    if search_query:
        return OllamaEmbeddings(model = embeddings, model_kwargs = {"prompt": "search_query:"})
    return OllamaEmbeddings(model = embeddings)


def _gradio_history_to_langchain(history: list):
    '''
    Converts Gradio messages format to LangChain message objects. Skips empty assistant placeholders.

    Part of RAG Query
    '''
    def extract(content):
        if isinstance(content, str): return content
        if isinstance(content, list): return " ".join(block.get("text", "") if isinstance(block, dict) else str(block) for block in content)
        return str(content)

    messages = []
    for item in history:
        content = extract(item["content"])
        if not content: continue
        if item["role"] == "user": messages.append(HumanMessage(content = content))
        elif item["role"] == "assistant": messages.append(AIMessage(content = content))

    return messages


def human_collection(collection:str):
    '''
    Turns the ascii collection into a human readable form.
    '''
    try:
        human_name = "".join(chr(int(code)) for code in collection.split("_"))
    
    except Exception as e:
        logger.warning(f"Error converting ASCII collection name to Human readable | {collection} | {type(e)} | {e}")
        return collection
    
    return human_name


def _load_document(file_path, DocLoader):
    '''
    Generic document loader function.

    Part of RAG Input
    '''
    if DocLoader is None:
        logger.warning(f"Document loader is set to None | file {file_path}")
        return None

    loader = DocLoader(file_path)
    document = loader.load()

    if not document:
        logger.warning(f"Loaded document returned None | file {file_path}")
        return None
    
    return document


def load_documents(file, hr_collection, embed_model, lang_model,
                   tags: list | None = None, chunk_size = 512, chunk_overlap = 50, chunk_batch = 50, chunk_sum = 10, 
                   *args, **kwargs):
    '''
    Loads the document from the input path, then add it to the database.

    Part of RAG Input
    '''
    logger.info(f"Loading file | {file} | {hr_collection} | {embed_model} | C Size: {chunk_size} | C Overlap: {chunk_overlap}")
    
    ascii_collection = _clean_collection(hr_collection)
    if isinstance(file, str): file = pathlib.Path(file)
    title = file.stem  # this is to ensure that there is a title being added to the metadata
    document = None
    chunks = None
    DocLoader = None

    suffix_map = {
        ".csv": UnstructuredCSVLoader, 
        ".doc": UnstructuredWordDocumentLoader,
        ".docx": UnstructuredWordDocumentLoader, 
        ".epub": UnstructuredEPubLoader, 
        ".md": UnstructuredMarkdownLoader,
        ".pdf": PDFPlumberLoader, 
        ".txt": TextLoader, 
                  }
    DocLoader = suffix_map.get(file.suffix)

    if DocLoader is None:
        logger.error(f"Unsupported file type | {file.suffix}")
        return f"Error: Unsupported file type: {file.suffix}"
    
    logger.info(f"Document loader set to {DocLoader.__name__}")
            
    try:
       document = _load_document(file, DocLoader)
    except Exception as e:
        logger.error(f"Failed to load document | {file} | {type(e)} | {e}")
        return f"Error loading file: {type(e)}"
    
    if document is None:
        logger.warning(f"Document loaded but was empty | {file}")

    try:
        chunks = _create_chunks(document, chunk_size, chunk_overlap)
    except Exception as e:
        logger.error(f"Failed to create chunks | {type(e)} | {e}")
        return f"Error chunking document: {type(e)}"
    
    if chunks is None:
        logger.warning(f"No chunks created from document | {file}")
        return "Error: no chunks could be created"
    
    chunks = _enrich_chunk_metadata(chunks, game_system = hr_collection)
    if tags:
        user_tags = set(tags) if isinstance(tags, list) else {tags}
        for chunk in chunks:
            existing = set(chunk.metadata.get("tags", "").split(",")) - {""}
            chunk.metadata["tags"] = ",".join(sorted(existing | user_tags))
    
    logger.info(f"Created {len(chunks)} chunks from {file.name} | Enriched with automated metadata")

    
    summary = _generate_section_summary(chunks, lang_model, chunk_sum)

    try:
        _load_to_Chroma(summary, ascii_collection, embed_model, add_ids = False, title = title, batch_size = chunk_batch)
        logger.info(f"Successfully loaded {file.name} summary | {hr_collection}")
    except Exception as e:
        logger.error(f"Failed to load summary to Chroma | {type(e)} | {e}")
        return f" Error writing summary to database: {e}"

    try:
        _load_to_Chroma(chunks, ascii_collection, embed_model, tags = tags or [], title = title, batch_size = chunk_batch)
        logger.info(f"Successfully loaded {file.name} | {hr_collection} | {tags}")
        return f"Successfully added {len(chunks)} chunks | {file.name} | {hr_collection}."
    except Exception as e:
        logger.error(f"Failed to load chunks to Chroma | {type(e)} | {e}")
        return f"Error writing to database: {e}"



def _load_to_Chroma(chunks, collection, embed_model, 
                    add_ids: bool = True, batch_size = 50, tags: list | None = None, title: str | None = None,
                    *args, **kwargs):
    '''
    Loads the documents to a Chroma DB

    Part of RAG Input
    '''
    hr_collection = human_collection(collection)
    db = Chroma(persist_directory = str(chroma_database_dir), embedding_function = _get_embeddings(embed_model), collection_name = collection)  # get embeddings here never uses the search doc prefix. Think about turning that on at some point.

    if add_ids: chunks = _metadata_IDs(chunks)

    existing_items = db.get(include = [])
    existing_ids = set(existing_items["ids"])

    # new_chunks = [c for c in chunks if c.metadata["id"] not in existing_ids]
    new_chunks = []
    for chunk in chunks:
        if chunk.metadata["id"] not in existing_ids:
            t = chunk.metadata.get("Title", None)
            if t is None: chunk.metadata["Title"] = title  # this is to make sure that the document/chunk has a title.
            # chunk.metadata["game_system"] = hr_collection
            # chunk.metadata["tags"] = ",".join(tags) if tags else ""  # chormaDB doesn't allow lists to be part of the metadata, it has to be a string.
            new_chunks.append(chunk)

    if not new_chunks:
        logger.info(f"No new chunks to add to {hr_collection}")
        return
    
    logger.info(f"Adding {len(new_chunks)} new chunks | Collection: {hr_collection} | Batches: {batch_size} ")

    for i in range(0, len(new_chunks), batch_size):
        batch = new_chunks[i: i + batch_size]
        batch_ids = [c.metadata["id"] for c in batch]
        try:
            db.add_documents(batch, ids = batch_ids)
            logger.info(f"Batch {i // batch_size + 1}/{-(-len(new_chunks)//batch_size)} complete | {min(i + batch_size, len(new_chunks))}/{len(new_chunks)} chunks added")
        except Exception as e:
            logger.error(f"Falied on batch {i//batch_size + 1} | {type(e)} | {e}")
            raise


def _looks_like_stat_block(text: str) -> bool:
    '''
    Tries to identify if the input context is a stat block or not.

    Currently this is hard coded, but much like the keywords, this would be better moved to an external file for importing. However because it's so specific to games, this will require a lot of research to find the right way to represent blocks. For example Rifts does something completely different and does not fit in this nicely.
    '''
    import re
    # common stat block patterns across RPG systems
    patterns = [
        r'\b(STR|DEX|CON|INT|WIS|CHA)\s*:?\s*\d+',  # D&D attributes
        r'\b(BOD|AGI|REA|STR|WIL|LOG|INT|CHA)\s*:?\s*\d+',  # Shadowrun
        r'\bAC\s*:?\s*\d+',  # armor class
        r'\bHP\s*:?\s*\d+',  # hit points
        r'\bCR\s*:?\s*[\d/]+',  # challenge rating
        r'\bINIT\s*:?\s*[+-]?\d+',  # initiative
    ]
    matches = sum(1 for p in patterns if re.search(p, text, re.IGNORECASE))
    return matches >= 2  # at least two stat indicators



def _looks_like_table(text: str) -> bool:
    '''
    Tries to determine if the content is a table or not.
    '''
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    if len(lines) < 3:
        return False
    
    pipe_lines = sum(1 for l in lines if "|" in l)
    tab_lines = sum(1 for l in lines if "\t" in l)
    short_lines = sum(1 for l in lines if len(l) < 60)

    if pipe_lines / len(lines) > 0.5:
        return True
    if tab_lines / len(lines) > 0.5:
        return True
    if short_lines / len(lines) > 0.7 and len(lines) > 5:
        return True
    
    return False


def _metadata_IDs(chunks, summary: bool = False, *args, **kwargs):
    '''
    Assigns a new metadata ID to the item. The metadata tag is: source document: page: chunk index. The chunk index for each document goes from [0, max chunks].
    
    Part of RAG Input
    '''
    last_page_id = None
    current_chunk_index = 0

    # there could be something here, like checking to see if the metadata was properly added, but I'm not sure. I might be logging things to log them.
    for chunk in chunks:
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        current_page_id = f"{source}:{page}"

        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0

        chunk_id = f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id

        chunk.metadata["id"] = chunk_id

    return chunks    


def _merge_retrievers(retrievers):
    '''
    MergeRetrievers is a legacy piece of code, and the new one is apparently also flagged to be moved to legacy. So,
    because I need to merge a bunch of responses into one, this is the custom way to do this.
    '''
    def retrieve(query: str):
        seen = set()
        merged = []
        for retriever in retrievers:
            docs = retriever.invoke(query)
            for doc in docs:
                if doc.page_content not in seen:
                    seen.add(doc.page_content)
                    merged.append(doc)
        return merged
    
    return RunnableLambda(retrieve)

# # Because I'm going to change this to query routing, I'm debating how much this needs to be logged right now.
# # Get response time for the query. Time it up to the last for loop
# def query_rag(message: str, history: list, collection: str, model: str, embeddings: str, 
#               *args, **kwargs): #type: ignore
#     '''
#     Streaming RAG query. Yields response chunks.
#     Uses a single prompt that retrieves context but instructs the LLM
#     to ignore it if irrelevant — handles both rules and general questions.

#     Part of RAG Query
#     '''
#     # Check how many responses are returned.
#     retriever = _get_retriever(collection, embeddings)
#     llm = ChatOllama(model=model)

#     prompt = ChatPromptTemplate.from_messages([
#         ("system", """
         
#          Rulebook context has been retrieved and is provided below. Use it if it is 
#          relevant to the question. If it is not relevant to the question, ignore it 
#          entirely and answer conversationally from your own knowledge, but state 
#          that you cannot find relevant information from the retrieved database.

#         Retrieved context:
#         {context}"""),
#         MessagesPlaceholder(variable_name="history"),
#         ("human", "{question}")
#         ])

#     lc_history = _gradio_history_to_langchain(history)

#     chain = (
#         {
#             "context": (lambda x: x["question"]) | retriever | _format_docs,
#             "question": lambda x: x["question"],
#             "history": lambda x: x["history"]
#         }
#         | prompt
#         | llm
#         | StrOutputParser()
#     )

#     response = ""
#     for chunk in chain.stream({"question": message, "history": lc_history}):
#         response += chunk
#         yield response


def query_rag_routed(message: str, history: list, lang_model: str, embed_model: str, 
                     collection: str | None = None, tags: list[str] | None = None, k: int = 10, score_threshold: float = 0.3,
                     *args, **kwargs):
    '''
    Routes the question to the correct response, or at least the response that matches the question.
    '''
    if isinstance(tags, str): 
        logger.warning(f"Tags had a string input | {tags} | Setting to NONE")
        tags = None

    from __tech_fn import load_tags
    available_tags = load_tags()
    rule_systems = find_collections()

    llm = ChatOllama(model = lang_model)

    classification, extracted_tags, extracted_rule_system  = _classify_and_tag(message, llm, rule_systems, available_tags)

    if classification != "RULES":
        yield from _direct_response(message, history, lang_model)
        return
    
    tags = tuple(set((tags or []) + extracted_tags)) or None

    logger.info(f"Classification | {classification} | {extracted_tags} | {extracted_rule_system} | {tags}")

    collections_to_search = None
    if collection and extracted_rule_system: collections_to_search = list(set([collection]) | set([extracted_rule_system]))
    elif collection: collections_to_search = [collection]
    elif extracted_rule_system: collections_to_search = [extracted_rule_system]
    else: collections_to_search = rule_systems

    if not collections_to_search:
        logger.warning("No collections came back, answering as GENERAL")
        yield from _direct_response(message, history, lang_model)
        return
    
    # if _query_wants_table(message):  # this will eventually add another 
    
    if len(collections_to_search) == 1:
        if _query_is_conceptual(message): retriever = _get_summary_retriever(collections_to_search[0], embed_model, k = k, tags = tags, score_threshold = score_threshold)
        else: retriever = _get_retriever(collections_to_search[0], embed_model, tags = tags, k = k, score_threshold = score_threshold)
    else:
        if _query_is_conceptual(message): retrievers = [_get_summary_retriever(c, embed_model, k = k, tags = tags, score_threshold = score_threshold) for c in collections_to_search] 
        else: retrievers = [_get_retriever(c, embed_model, tags = tags, k = k, score_threshold = score_threshold) for c in collections_to_search]
        retriever = _merge_retrievers(retrievers)
    
    logger.info(f"Generating query | {lang_model} | {embed_model}")
    
    yield from _rag_response(message, history, lang_model, retriever)


def _rag_response(message, history, lang_model, retriever):
    '''
    Shared RAG generation logic. Seperated so both query_rag and query_rag_routed can use it without duplication.

    However query_rag is not going to be used in the future, so this is instead a nice way to apply logging logic and find errros.
    '''
    llm = ChatOllama(model = lang_model)

    prompt = ChatPromptTemplate.from_messages([("system", """
                                                Rulebook context has been retrieved and is provided below. Use it if it is relevant to the question. 
                                                If it is, cite the page numbers and books the information is coming from. If it is not relevant to 
                                                the question, ignore it entirely and answer conversationally from your own knowledge, but state that 
                                                you cannot find relevant information from the retrieved database.
                                                
                                                Retrieved context:
                                                {context}
                                                """),
                                                MessagesPlaceholder(variable_name = "history"),
                                                ("human", "{question}")
                                                ])
    
    lc_history = _gradio_history_to_langchain(history)

    chain = (
        {
            "context": (lambda x: x["question"]) | retriever | _format_docs,
            "question": lambda x: x["question"],
            "history": lambda x: x["history"]
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    reponse = ""
    try:
        for chunk in chain.stream({"question": message, "history": lc_history}):
            reponse += chunk
            yield reponse
    except chromadb.errors.InvalidArgumentError as e:
        logger.critical(f"Unable to return message | check embeddeing model being used | {e}")
        return f"Error generating response"
    except Exception as e:
        logger.critical(f"Unable to return message | New Error | {type(e)} | {e}")
        return f"Error generating response"


def _query_is_conceptual(msg: str) -> bool:
    '''
    Determines if the query is conceptual and needs a broad overview.
    '''
    conceptual_signals = {"describe", "explain", "generally", "how does", "in general", "overview", "summary", "tell me about", "walk me through", "what is"}
    msg_lower = msg.lower()
    return any(signal in msg_lower for signal in conceptual_signals)



def _query_wants_table(msg: str) -> bool:
    '''
    Considers if the input query from the user is looking for tabular data.
    '''
    # Again, this might be better for something outside the script rather than hard coded, but I'm going to have to leave this here.
    table_signals = {"table", "chart", "list", "modifier", "cost", "price", "range", "damage", "stat", "attribute", "how much", "what is the", "roll for"}
    msg_lower = msg.lower()
    return any(signal in msg_lower for signal in table_signals)


def update_metadata(hr_collection, title, new_tags):
    '''
    Updates the metadata with new tags.

    This updates individual chunks, so it is theoretically possible to target specific chunks for different metadata.
    '''
    # new_tags = ",".join(new_tags)
    new_tags = set(new_tags)
    ascii_collection = _clean_collection(hr_collection)
    logger.info(f"Adding metadata | {title} | {new_tags} | {hr_collection} | {ascii_collection}")

    client = _get_client()
    local_collection = client.get_collection(str(ascii_collection))
    local_result = local_collection.get(where = {"Title": title}, include = ["metadatas"])  # the thing about python: You can do things like this in pandas but NOT here. You have to pull it from the variable itself

    if not local_result["ids"]:
        logger.warning(f"No Chunks found | {title} | {hr_collection}")
        return
    
    updated_metadatas = []

    for meta in local_result["metadatas"]:
        old_tags = set(meta.get("tags", "").split(",")) - {""} # this strips the empty string. Useful for if "tags" is empty.
        merged_tags = old_tags | new_tags
        updated_meta = {**meta, "tags": ",".join(sorted(merged_tags))}  # unpack, sort the list, and join the whole thing into a string
        updated_metadatas.append(updated_meta)

    local_collection.update(ids = local_result["ids"], metadatas =  updated_metadatas)

    logger.info(f"Finished updating metadata | {title} | {hr_collection} | {updated_meta}")




logger.info(f"Finished reading RAG Pipeline file @ (time to be implemented)")