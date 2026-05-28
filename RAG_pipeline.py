# import chromadb

# Something on this page is causing it to throw out warning and be slow to upload.

# from langchain_classic.prompts import ChatPromptTemplate  # need to figure out how to get around this -> use langchain_core instead of classic

from langchain_chroma import Chroma
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import UnstructuredCSVLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableBranch, RunnableLambda, RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter
# import os

# import pandas
import pathlib
cwd = pathlib.Path.cwd()
chroma_database_dir = cwd / "DB_of_Holding"

print("Finished loading RAG Pipeline")

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


def retrieve_collection(collection:str):
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


def _get_embeddings():
    '''
    Currently this is hard coded to only use the qwen3 model embeddings.
    '''
    return OllamaEmbeddings(model = "qwen3-embedding:4b")


def load_documents(file, collection, chunk_size, chunk_overlap, *args, **kwargs):
    '''
    Loads the document from the input path, then add it to the database.
    '''
    collection = _clean_collection(collection)
    if isinstance(file, str): file = pathlib.Path(file)
    print(f"#####\nChunk Size = {chunk_size}, {type(chunk_size)}\nOverlap = {chunk_overlap}, {type(chunk_overlap)}\n#####")
    document = None
    chunks = None
    DocLoader = None

    if file.suffix == ".pdf":
        DocLoader = PDFPlumberLoader
    elif file.suffix == ".txt":
        DocLoader = TextLoader
    elif file.suffix == ".csv":
        DocLoader = UnstructuredCSVLoader  # this one might need some more testing, as csv files have headers and those might need to be read in properly.
    
    document = _load_document(file, DocLoader)

    if document is not None:
        chunks = _create_chunks(document, chunk_size, chunk_overlap)

    if chunks is not None:
        _load_to_Chroma(chunks, collection)


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


def _load_to_Chroma(chunks, collection, *args, **kwargs):
    '''
    Loads the documents to a Chroma DB
    '''
    db = Chroma(persist_directory = str(chroma_database_dir), embedding_function = _get_embeddings(), collection_name = collection)

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


def _get_retriever(collection: str, k: int = 10):
    '''
    Returns a retriever for the given collection name.
    '''
    ascii_collection = _clean_collection(collection)
    db = Chroma(persist_directory = str(chroma_database_dir), embedding_function = _get_embeddings(), collection_name = ascii_collection)
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
    messages = []
    for item in history:
        if item["role"] == "user": messages.append(HumanMessage(content = item["content"]))
        elif item["role"] == "assistant" and item["content"]: messages.append(AIMessage(content = item["content"]))

    return messages

def query_rag(message: str, history: list, collection: str, system_content: str, model: str):
    '''
    Streaming RAG query. Yields response chunks.
    Uses a single prompt that retrieves context but instructs the LLM
    to ignore it if irrelevant — handles both rules and general questions.
    '''
    retriever = _get_retriever(collection)
    llm = ChatOllama(model=model)

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_content + """

Rulebook context has been retrieved and is provided below. Use it if it is 
relevant to the question. If it is not relevant to the question, ignore it 
entirely and answer conversationally from your own knowledge.

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