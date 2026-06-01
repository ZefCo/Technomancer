import logging
logger = logging.getLogger(__name__)

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

import pathlib

cwd = pathlib.Path.cwd()
chroma_database_dir = cwd / "DB_of_Holding"

_chroma_client = None

 # --------------------------------- Other --------------------------------- #
# -------------------------- Waiting to be sorted -----=--------------------- #

def _get_client():
    '''
    '''
    global _chroma_client
    if _chroma_client is None: _chroma_client = chromadb.PersistentClient(path = str(chroma_database_dir))
    return _chroma_client

def _clean_collection(collection: str):
    '''
    Turns the collection into a valid name, which will be ascii values. Each ascii value is separated by an _
    
    Because it turns it into a series of ascii values in the form x_x_x_x where each x is an ascii value of 2-3 numbers, this means the input string must be less than 128 characters long. That's OK. I doubt anyone will want to title a collection that long.
    '''
    # ascii_collection = f"{ord(collection[0])}"
    # for c in collection[1:len(collection)]:
    #     ascii_collection = f"{ascii_collection}_{ord(c)}"

    ascii_collection = "_".join(str(ord(c)) for c in collection)

    if len(ascii_collection) < 3:
        for _ in range(3 - len(ascii_collection)): ascii_collection = ascii_collection + "_{ord(z)}"

    if len(ascii_collection) > 512:
        ascii_collection = ascii_collection[:512]

    return ascii_collection


def create_collection(collection: str):
    '''
    Creates an empty collection if that collection name is not already in use.
    '''
    collection = _clean_collection(collection)
    client = _get_client()
    _ = client.get_or_create_collection(name = collection)


def delete_document(collection, metadata):
    '''
    Deletes a document in the database. Does so by finding everything with matching metadata and deleting it. Searches on the source of the data, which is hopefully reliable.

    Again, use the human readable version of the collection name.
    '''
    client = _get_client()
    
    collection = _clean_collection(collection)
    local_collection = client.get_collection(str(collection))
    
    local_collection.delete(where = {"Title": metadata})



def find_collections():
    '''
    Returns all collections in the database.

    Checks to see if anything is present. If there is, it will clean up the collection name into a human readable format and return it. If there is nothing, just returns an empty list.
    '''
    client = _get_client()
    collections = client.list_collections()
    if collections: collections = [human_collection(collection.name) for collection in collections]

    return collections


def find_documents(collection):
    '''
    Finds all available documents in a given collection. Feed in the human readable collection title.
    '''
    titles = set()
    collection = _clean_collection(collection)

    client = _get_client()
    local_collection = client.get_collection(str(collection))
    results: dict = local_collection.get() # type: ignore  There's an error here that shouldn't be an error. It works fine.

    for item in results["metadatas"]:
        title = item.get("Title", None)
        if title: titles.add(title)
        else: titles.add(pathlib.Path(item["source"]).stem)  # this is in case the title doesn't have a Title.
    
    return list(titles)


def human_collection(collection:str):
    '''
    Turns the ascii collection into a human readable form.
    '''
    return "".join(chr(int(code)) for code in collection.split("_"))



### ---------------------------------- RAG INPUT ---------------------------------- ###


def _create_chunks(document, chunk_size, chunk_overlap, *args, **kwargs):
    '''
    Chunks the document
    '''
    split_text = RecursiveCharacterTextSplitter(chunk_size = chunk_size, chunk_overlap = chunk_overlap)

    chunks = split_text.split_documents(document)

    if not chunks:
        return None
    
    return chunks


@lru_cache(maxsize=4)
def _get_embeddings(embeddings):
    '''
    Currently this is hard coded to only use the qwen3 model embeddings.
    '''
    return OllamaEmbeddings(model = embeddings)


def load_documents(file, collection, chunk_size, chunk_overlap, embeddings,
                   *args, **kwargs):
    '''
    Loads the document from the input path, then add it to the database.
    '''
    collection = _clean_collection(collection)
    if isinstance(file, str): file = pathlib.Path(file)
    document = None
    chunks = None
    DocLoader = None

    if file.suffix == ".pdf":
        DocLoader = PDFPlumberLoader
    elif file.suffix == ".txt":
        DocLoader = TextLoader
    elif file.suffix == ".csv":
        DocLoader = UnstructuredCSVLoader  # this one might need some more testing, as csv files have headers and those might need to be read in properly.
    elif file.suffix == ".epub":
        DocLoader = UnstructuredEPubLoader
    
    print(f"#####\nIngesting {file} into {collection}\nChunk Size = {chunk_size}\nOverlap = {chunk_overlap}\n#####")
    
    
    document = _load_document(file, DocLoader)

    if document is not None:
        chunks = _create_chunks(document, chunk_size, chunk_overlap)

    if chunks is not None:
        print("Loading to Database of Holding")
        _load_to_Chroma(chunks, collection, embeddings)
    else:
        print("Failed to get anything to load to Database of Holding")
    print("###Finished###")


def _load_pdf(file_path):
    '''
    '''
    loader = PDFPlumberLoader(file_path)
    document = loader.load()

    if not document:
        return None
    
    return document


def _load_txt(file_path):
    '''
    '''
    loader = TextLoader(file_path)
    document = loader.load()

    if not document:
        return None
    
    return document


def _load_document(file_path, DocLoader):
    '''
    Generic document loader function.
    '''
    if DocLoader is None:
        return None

    loader = DocLoader(file_path)
    document = loader.load()

    if not document:
        return None
    
    return document


def _load_to_Chroma(chunks, collection, embeddings, 
                    *args, **kwargs):
    '''
    Loads the documents to a Chroma DB
    '''
    db = Chroma(persist_directory = str(chroma_database_dir), embedding_function = _get_embeddings(embeddings), collection_name = collection)

    chunks = _metadata_IDs(chunks)
    
    existing_items = db.get(include = [])
    existing_ids = set(existing_items["ids"])
    # print(f"Number of existing documents in the Database: {len(existing_ids)}")

    new_chunks = []
    for chunk in chunks:
        if chunk.metadata["id"] not in existing_ids:
            new_chunks.append(chunk)

    if len(new_chunks):
        # print(f"Adding {len(new_chunks)} new documents to database")
        new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
        db.add_documents(new_chunks, ids = new_chunk_ids)

    # else:
    #     print("No new documents to add")



def _metadata_IDs(chunks, *args, **kwargs):
    '''
    Assigns a new metadata ID to the item. The metadata tag is: source document: page: chunk index. The chunk index for each document goes from [0, max chunks].
    '''
    last_page_id = None
    current_chunk_index = 0

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


### ---------------------------------- RAG QUERY ---------------------------------- ###


@lru_cache(maxsize = 8)
def _get_retriever(collection: str, embeddings: str, k: int = 10):
    '''
    Returns a retriever for the given collection name.
    '''
    ascii_collection = _clean_collection(collection)
    db = Chroma(persist_directory = str(chroma_database_dir), embedding_function = _get_embeddings(embeddings), collection_name = ascii_collection)
    return db.as_retriever(search_kwargs = {"k": k})


def _format_docs(docs):
    '''
    Formats retrieved documents into a single context string. Includes source metadata so the LLM can cite pages.
    '''
    formatted = []
    for doc in docs:
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "?")
        formatted.append(f"[Source: {source}, Page: {page}]\n{doc.page_content}")
    return "\n\n".join(formatted)


def _gradio_history_to_langchain(history: list):
    '''
    Converts Gradio messages format to LangChain message objects. Skips empty assistant placeholders.
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


def query_rag(message: str, history: list, collection: str, model: str, embeddings: str, system_content: str = "", 
              *args, **kwargs): #type: ignore
    '''
    Streaming RAG query. Yields response chunks.
    Uses a single prompt that retrieves context but instructs the LLM
    to ignore it if irrelevant — handles both rules and general questions.
    '''
    retriever = _get_retriever(collection, embeddings)
    llm = ChatOllama(model=model)

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_content + """
         
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


print("Finished loading RAG Pipeline")
