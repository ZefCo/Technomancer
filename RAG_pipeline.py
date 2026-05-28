# import chromadb

# from langchain_classic.prompts import ChatPromptTemplate  # need to figure out how to get around this
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PDFPlumberLoader
# from langchain_community.llms.ollama import Ollama
from langchain_chroma import Chroma
from langchain_community.embeddings import OllamaEmbeddings

# import os

# import pandas
import pathlib
cwd = pathlib.Path.cwd()
chroma_database_dir = cwd / "DB_of_Holding"
import re


# class RAG_input:
#     '''
#     This is the RAG pipeline from the UI and Ollama to the ChromaDB using LangChain.
#     '''

#     def __init__(self):
#         pass

def _clean_collection(collection: str):
    '''
    Turns the collection into a valid name, which will be ascii values. Each ascii value is separated by an _
    
    Because it turns it into a series of ascii values in the form x_x_x_x where each x is an ascii value of 2-3 numbers, this means the input string must be less than 128 characters long. That's OK. I doubt anyone will want to title a collection that long.
    '''
    ascii_collection = f"{ord(collection[0])}"
    for c in collection[1:len(collection)]:
        ascii_collection = f"{ascii_collection}_{ord(c)}"

    if len(ascii_collection) < 3:
        for _ in range(3 - len(ascii_collection)): ascii_collection = ascii_collection + "_{ord(z)}"

    if len(ascii_collection) > 512:
        ascii_collection = ascii_collection[:512]

    return ascii_collection


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
    # for file in self._yield_documents():
    if file.suffix == ".pdf":
        document = _load_pdf(file)
    
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

