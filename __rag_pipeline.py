from datetime import datetime
import pathlib
from os.path import basename
from __log_fn import setup_logs
import logging
logger = logging.getLogger(__name__)
logger.info(f"Reading RAG Pipeline file")



import chromadb

from functools import lru_cache

from langchain_chroma import Chroma

from langchain_community.document_loaders import PDFPlumberLoader, TextLoader, UnstructuredCSVLoader, UnstructuredEPubLoader

from langchain_ollama import OllamaEmbeddings  # this should hopefully get rid of that warning about depreciation
from langchain_ollama import ChatOllama

from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableBranch, RunnableLambda, RunnablePassthrough

from langchain_text_splitters import RecursiveCharacterTextSplitter

cwd = pathlib.Path.cwd()
chroma_database_dir = cwd / "DB_of_Holding"

_chroma_client = None


def _clean_collection(collection: str):
    '''
    Turns the collection into a valid name, which will be ascii values. Each ascii value is separated by an _
    
    Because it turns it into a series of ascii values in the form x_x_x_x where each x is an ascii value of 2-3 numbers, this means the input string must be less than 128 characters long. That's OK. I doubt anyone will want to title a collection that long.
    '''
    # ascii_collection = f"{ord(collection[0])}"
    # for c in collection[1:len(collection)]:
    #     ascii_collection = f"{ascii_collection}_{ord(c)}"

    ascii_collection = "_".join(str(ord(c)) for c in collection)

    logger.info(f"Convertion Str to ASCII | {collection} | {ascii_collection}")

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


def delete_document(collection, metadata):
    '''
    Deletes a document in the database. Does so by finding everything with matching metadata and deleting it. Searches on the source of the data, which is hopefully reliable.

    Again, use the human readable version of the collection name.
    '''
    client = _get_client()
    
    collection = _clean_collection(collection)
    local_collection = client.get_collection(str(collection))
    
    try:
        local_collection.delete(where = {"Title": metadata})
    except Exception as e:
        logger.critical(f"Error deleting chunks for the document with metadata {metadata}")


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
def find_documents(collection):
    '''
    Finds all available documents in a given collection. Feed in the human readable collection title.
    '''
    titles = set()
    collection = _clean_collection(collection)

    client = _get_client()
    local_collection = client.get_collection(str(collection))
    results: dict = local_collection.get() # type: ignore  There's an error here that shouldn't be an error. It works fine.

    items = 0
    for item in results["metadatas"]:
        title = item.get("Title", None)
        try:
            if title: titles.add(title)
            else: titles.add(pathlib.Path(item["source"]).stem)  # this is in case the title doesn't have a Title.
        except Exception as e:
            logger.warning(f"Cannont find title or source in item retreived from collection {collection} | Error type {type(e)} | {e}")
        else:
            items += 1
    logger.info(f"Successfully found {items} documents in collection {collection}")
    
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


# Log when this is called and creates a new client.
def _get_client():
    '''
    Connects to the client
    '''
    # https://stackoverflow.com/questions/77134962/connecttimeout-error-when-connecting-to-a-chromadb-client-that-is-hosted-on-azur
    # Add connection issues to the client. Send those to the logs
    global _chroma_client
    if _chroma_client is None: _chroma_client = chromadb.PersistentClient(path = str(chroma_database_dir))
    return _chroma_client


@lru_cache(maxsize=4)
def _get_embeddings(embeddings):
    '''
    Currently this is hard coded to only use the qwen3 model embeddings.

    Part of RAG Input
    '''
    return OllamaEmbeddings(model = embeddings)


@lru_cache(maxsize = 8)
def _get_retriever(collection: str, embeddings: str, k: int = 10):
    '''
    Returns a retriever for the given collection name.

    Part of RAG Query
    '''
    if not collection: collection = "Generic"
    ascii_collection = _clean_collection(collection)
    db = Chroma(persist_directory = str(chroma_database_dir), embedding_function = _get_embeddings(embeddings), collection_name = ascii_collection)
    return db.as_retriever(search_kwargs = {"k": k})


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
    return "".join(chr(int(code)) for code in collection.split("_"))


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


def load_documents(file, collection, chunk_size, chunk_overlap, embeddings,
                   *args, **kwargs):
    '''
    Loads the document from the input path, then add it to the database.

    Part of RAG Input
    '''
    error_msg = ""
    collection = _clean_collection(collection)
    if isinstance(file, str): file = pathlib.Path(file)
    document = None
    chunks = None
    DocLoader = None

    if file.suffix == ".pdf":
        DocLoader = PDFPlumberLoader
        logger.info(f"Document Loader function set to PDF")
    elif file.suffix == ".txt":
        DocLoader = TextLoader
        logger.info(f"Document Loader function set to Text")
    elif file.suffix == ".csv":
        DocLoader = UnstructuredCSVLoader  # this one might need some more testing, as csv files have headers and those might need to be read in properly.
        logger.info(f"Document Loader function set to Unstructured CSV")
    elif file.suffix == ".epub":
        DocLoader = UnstructuredEPubLoader
        logger.info(f"Document Loader function set to Unstructured Epub")
    else:
        extention = file.suffix
        logger.warning(f"Unsuported file type | File extension {extention}")
            
    document = _load_document(file, DocLoader)

    # if document is not None:
    if document:
        chunks = _create_chunks(document, chunk_size, chunk_overlap)

    # if chunks is not None:
    if chunks:
        _load_to_Chroma(chunks, collection, embeddings)
    else:
        logger.error(f"The document was not able to be loaded")
        error_msg = "Failed load document to the database, check logs for possible reasons."

    return error_msg


# def _load_pdf(file_path):
#     '''
#     No longer used

#     Part of RAG Input
#     '''
#     loader = PDFPlumberLoader(file_path)
#     document = loader.load()

#     if not document:
#         return None
    
#     return document


# def _load_txt(file_path):
#     '''
#     No longer used

#     Part of RAG Input
#     '''
#     loader = TextLoader(file_path)
#     document = loader.load()

#     if not document:
#         return None
    
#     return document


def _load_to_Chroma(chunks, collection, embeddings, 
                    *args, **kwargs):
    '''
    Loads the documents to a Chroma DB

    Part of RAG Input
    '''
    db = Chroma(persist_directory = str(chroma_database_dir), embedding_function = _get_embeddings(embeddings), collection_name = collection)
    # update this to tell the log if this is successful or not, and with which embedding was used.

    chunks = _metadata_IDs(chunks)
    
    existing_items = db.get(include = [])
    existing_ids = set(existing_items["ids"])
    # print(f"Number of existing documents in the Database: {len(existing_ids)}")

    # upadte this to see how many items are going to be added. Make sure it reminds the log reader how many items were supposed to be added.
    new_chunks = []
    for chunk in chunks:
        if chunk.metadata["id"] not in existing_ids:
            new_chunks.append(chunk)

    # Update this to make sure those chcunks were in fact added.
    if len(new_chunks):
        # print(f"Adding {len(new_chunks)} new documents to database")
        new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
        db.add_documents(new_chunks, ids = new_chunk_ids)

    # else:
    #     print("No new documents to add")


def _metadata_IDs(chunks, *args, **kwargs):
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


# Because I'm going to change this to query routing, I'm debating how much this needs to be logged right now.
# Get response time for the query. Time it up to the last for loop
def query_rag(message: str, history: list, collection: str, model: str, embeddings: str, 
              *args, **kwargs): #type: ignore
    '''
    Streaming RAG query. Yields response chunks.
    Uses a single prompt that retrieves context but instructs the LLM
    to ignore it if irrelevant — handles both rules and general questions.

    Part of RAG Query
    '''
    # Check how many responses are returned.
    retriever = _get_retriever(collection, embeddings)
    llm = ChatOllama(model=model)

    prompt = ChatPromptTemplate.from_messages([
        ("system", """
         
         Rulebook context has been retrieved and is provided below. Use it if it is 
         relevant to the question. If it is not relevant to the question, ignore it 
         entirely and answer conversationally from your own knowledge, but state 
         that you cannot find relevant information from the retrieved database.

        Retrieved context:
        {context}"""),
        MessagesPlaceholder(variable_name="history"),
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

    response = ""
    for chunk in chain.stream({"question": message, "history": lc_history}):
        response += chunk
        yield response


logger.info(f"Finished reading RAG Pipeline file")