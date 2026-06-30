from datetime import datetime
import pathlib
import logging
logger = logging.getLogger(__name__)
logger.info(f"Reading RAG Pipeline file @ (time to be implemented)")


import base64

import chromadb

from functools import lru_cache

import gradio as gr

import io

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

from __log_context import set_current_user

import ollama
import os

import pdfplumber
from pdfplumber.page import Page as PageClass

# from __tech_fn import import_settings   # this might get me in trouble in the long run, and I may need to have an import settings function here.
import toml

from rank_bm25 import BM25Okapi

cwd = pathlib.Path.cwd()
# chroma_database_dir = cwd / "DB_of_Holding"

with open(cwd / "Settings" / "Settings.toml", "r") as file:
    SETTINGS = toml.load(file)
    user_CHROMA_PORT: int = int(SETTINGS["ports"]["chroma"])
    user_OLLAMA_HOST: int = int(SETTINGS["ports"]["ollama"])
    QUALITY_THRESHOLDS = SETTINGS["QUALITY_THRESHOLDS"]
del SETTINGS  # all the settings don't need to be held here.
     

CHROMA_HOST = os.environ.get("CHROMA_HOST", "localhost")
CHROMA_PORT = int(os.environ.get("CHROMA_PORT", user_CHROMA_PORT))
_chroma_client = None
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", f"http://localhost:{user_OLLAMA_HOST}")


with open(cwd / "Settings" / "EnrichTags.toml", "r") as file:
    enrich_keywords = toml.load(file)
    enrich_keywords: dict = enrich_keywords["enrich_tags"]


def _classify_and_tag(msg: str, llm, rules_systems: list[str], available_tags: list[str]):
    '''
    Classifies the message and adds the metadata tags to the LMM.

    Tweak this to handle the possibility of different editions. Let the LLM output a list of rules and it can choose the most probable one later.
    '''
    prompt = ChatPromptTemplate.from_messages([
        ("system", f"""Analyze the user query and respond with exactly three lines and nothing else. Do not add preambles, extra lines, or anything else.
         Line 1: Either RULES or GENERAL
         Line 2: Comma-separated relevant tags from this list: {", ".join(available_tags)}, or None
         Line 3: the name of the rule system use from this list: {", ".join(rules_systems)}, or NONE

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


def _clean_double_chars(text: str) -> str:
    '''
    Cleans text with duplicate characters: TTRROOIIKKAA -> TROIKA
    '''
    def fix_word(word):
        if len(word) >= 4 and all (word[i] == word[i + 1] for i in range(0, len(word) - 1)):
            return word[::2]  # returns every other character
        return word
    
    words = text.split()
    fixed = [fix_word(w) for w in words]
    return " ".join(fixed)


def connect_to_database(host: str, port: int):
    '''
    This connects to the database.
    '''
    # client = chromadb.HttpClient(host='localhost', port=8000)
    # print(client.heartbeat())  # <- makes sure the client is working.


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

    # logger.critical(f"{type(chunks)}\n{chunks}")
    
    return chunks


# Add to log that this is happening
def create_collection(hr_collection: str):
    '''
    Creates an empty collection if that collection name is not already in use.
    '''
    logger.info(f"Creating new Collection in Database")
    ascii_collection = _clean_collection(hr_collection)
    client = _get_client()
    try:
        _ = client.get_or_create_collection(name = ascii_collection, metadata={"hnsw:space": "cosine"})
    except Exception as e:
        logger.critical(f"Error creating collection {hr_collection} in database | {ascii_collection}")
    else:
        logger.info(f"Successfully created {hr_collection} in database")



def delete_chunk(id: str, hr_collection: str, request: gr.Request):
    '''
    Deletes a specific chunk.

    I've been using the terms as follows:

    Collection: a collection of documents
    Document: a single file that is ingested
    Chunk: a portion of that document

    Chroma doesn't use chunks, as the document can be very long.
    '''
    set_current_user(request.username)
    client = _get_client()

    ascii_collection = _clean_collection(hr_collection)
    logger.warning(f"Deleting chunk {id} | {hr_collection}")

    local_collection = client.get_collection(name = ascii_collection)
    local_collection.delete(ids = [id])


    
def delete_collection(hr_collection, request: gr.Request,
                      nuclear_option: bool = False):
    '''
    Deletes an entire collection of documents.
    '''
    set_current_user(request.username)
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

    return None



def delete_document(hr_collection, metadata, request: gr.Request):
    '''
    Deletes a document in the database. Does so by finding everything with matching metadata and deleting it. Searches on the source of the data, which is hopefully reliable.

    Again, use the human readable version of the collection name.

    Metadata is really the title. The variable is named as such because it's pulling from the metadata to get the title.
    '''
    set_current_user(request.username)
    client = _get_client()
    
    ascii_collection = _clean_collection(hr_collection)
    logger.info(f"Deleting {metadata} | {hr_collection} | {ascii_collection}")
    local_ascii_collection = client.get_collection(str(ascii_collection))
    
    try:
        local_ascii_collection.delete(where = {"Title": metadata})
    except Exception as e:
        logger.critical(f"Error deleting chunks for the document with metadata {metadata} | {hr_collection} | {type(e)} | {e}")
    else:
        documents = find_documents(hr_collection, request)
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

    try:
        for chunk in completion:
            if "message" in chunk and "content" in chunk["message"]:
                response += chunk["message"]["content"]
                yield response
    except Exception as e:
        logger.critical(f"Unable to generate response | Type: {type(e)} | {e}")
        raise


def _enrich_chunk_metadata(chunks, game_system: str, embed_model: str):
    '''
    Tags specific documents with certain tags, making sections easier to identify later.
    '''
    # keywords: dict = import_setting("Keyword.yaml")
    # keywords: dict = enrich_keywords
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

        for term, tag_terms in enrich_keywords.items():
            if set(tag_terms) & set(text.split()):
                auto_tags.add(term)

        chunk.metadata["game_system"] = game_system
        chunk.metadata["auto_tags"] = fill_list(list(sorted(auto_tags)), game_system = game_system)  # ",".join(sorted(auto_tags))
        chunk.metadata["tags"] = fill_list(list(sorted(chunk.metadata["auto_tags"])))  # ",".join(sorted(auto_tags))  # I'm separating them like this so I can see what was added and what I added.
        chunk.metadata["embedding_used"] = embed_model

        # existing_tags = set(chunk.metadata.get("tags", "").split(",")) - {""}  # This is just to make sure that the set doesn't have any weird things in it
        # chunk.metadata["tags"] = fill_list(list(sorted(existing_tags)))  # ",".join(sorted(existing_tags | auto_tags))

    return chunks



def _extract_page_multimodal(page: PageClass, vision_model: str) -> str:
    '''
    Uses a vision LLM to extract the text and information of a PDF that has been converted to an image.
    This could, in theory, handle anything as the page is converted to an image and is then 'looked at' by the LLM.
    '''
    image_64b = _render_page_as_base64(page)
    prompt = """Transcribe all text from this page exactly as it reads, in natural reading order.

Rules:
- For two-column layouts: transcribe the left column completely, then the right column
- For tables: use markdown table format (| col1 | col2 |)
- For stat blocks or character sheets: preserve the label-value structure
- For sidebars or callout boxes: mark them with [SIDEBAR] at the start
- For headers and footers: skip them entirely
- Preserve section headings using ## and ### markdown
- Do not summarize, interpret, or add anything not on the page
- If a section is a form or character sheet with blank fields, transcribe the labels only"""

    response = ollama.chat(
        model=vision_model,
        messages=[{
            "role": "user",
            "content": prompt,
            "images": [image_64b]
        }]
    )
    
    return response["message"]["content"]



def find_chunk(hr_collection: str, ids: str, request: gr.Request):
    '''
    Gets all the data from the specified chunk
    '''
    # set_current_user(request.username)
    ascii_collection = _clean_collection(hr_collection)
    client = _get_client()
    local_collection = client.get_collection(str(ascii_collection))

    local_chunk: dict = local_collection.get(ids = [ids])

    # logger.critical(f"Local Chunk | {local_chunk} | {type(local_chunk)}")

    # I'm keeping this here as the project expands: things will be added or removed depending on the needs of the project
    # the local chunk is a dictionary with (* means things that will be used):
    # It's rather confusing. It's pulling out one chunk, but returning it as a list because I *could* return multiple chunks. I don't want to though.
    # Then there's how the data is organized. It's a series of lists, so we can pull the first index since it's only one chunk, but the data is scattered across multiple lists.

    # *ids -> the ids of the chunk
    # embeddings -> ?
    # metadatas -> a list with
        # source -> where it came from. Can be changed to Rule_System:Book_Title
        # auto_tags -> list
        # *page -> #
        # id -> str
        # *chunk_type -> string
        # Trapped -> bool
        # ModData -> datetime
        # *game_system -> string
        # Creator -> string
        # *total_pages -> #
        # *tags -> list
        # file_path -> str
        # *Title -> str
        # Producer -> str
        # *embedding_used -> str
        # source_pages -> str | int -> make sure it's a string
    # *documents -> list -> maybe only need the first index?
    # data -> ?
    # uris -> ?
    # what was included to find it

    # "angled_ratio": angled_ratio,
    # "doubled_ratio": doubled_ratio,
    # "word_count": word_count,
    # "ave_word_len": round(ave_word_len, 2),
    # "word_len_suspicious": word_len_suspicious,
    # "text_length": len(text),
    # "has_images_only": chunk_metadata.get("is_sparse", False) and chunk_metadata.get("has_images", False),

    logger.critical(f"{local_chunk['metadatas'][0]}")

    angled_score: float = local_chunk["metadatas"][0].get("angled_ratio", "?")
    auto_tags: list[str] = local_chunk["metadatas"][0].get("auto_tags", [])
    ave_word_score: float = local_chunk["metadatas"][0].get("ave_word_len", "?")
    chunk_data: str = local_chunk["documents"][0]
    chunk_tags: list[str] = local_chunk["metadatas"][0].get("tags", [])
    chunk_type: str = local_chunk["metadatas"][0].get("chunk_type", "?")
    doubled_score: float = local_chunk["metadatas"][0].get("doubled_ratio", "?")
    extraction_method: str = local_chunk["metadatas"][0].get("extraction_method", "?")
    has_images_bool: bool = local_chunk["metadatas"][0].get("has_images_only", "?")
    is_sparse_bool: bool = local_chunk["metadatas"][0].get("is_sparse", "?")
    page: int = local_chunk["metadatas"][0].get("page", "?")
    quality_pass: bool = local_chunk["metadatas"][0].get("quality_pass", "?")
    quality_score: float = local_chunk["metadatas"][0].get("quality_score", "?")
    source_string: str = local_chunk["metadatas"][0].get("source", "?")
    text_density_score: float = local_chunk["metadatas"][0].get("text_density", "?")
    text_len_score: int = local_chunk["metadatas"][0].get("text_length", "?")
    word_count_score: int = local_chunk["metadatas"][0].get("word_count", "?")
    word_len_score: bool = local_chunk["metadatas"][0].get("word_len_suspicious", "?")

    return (
        gr.Number(value = angled_score),
        gr.Dropdown(value = auto_tags),
        gr.Number(value = ave_word_score),
        gr.TextArea(value = chunk_data),
        gr.Dropdown(value = chunk_tags),
        gr.Textbox(value = chunk_type),
        gr.Number(value = doubled_score),
        gr.Textbox(value = extraction_method),
        gr.Textbox(value = str(has_images_bool)),
        gr.Textbox(value = str(is_sparse_bool)),
        gr.Number(value = page),
        gr.Textbox(value = str(quality_pass)),
        gr.Number(value = quality_score),
        gr.Textbox(value = source_string),
        gr.Number(value = text_density_score),
        gr.Number(value = text_len_score),
        gr.Number(value = word_count_score),
        gr.Textbox(value = str(word_len_score))
        )



def find_chunks(hr_collection: str, title: str, just_ids = True):
    '''
    Gets all the chunk ids for a given document.
    '''
    ascii_collection = _clean_collection(hr_collection)
    client = _get_client()
    local_collection = client.get_collection(str(ascii_collection))

    local_chunks = local_collection.get(where = {"Title": title})

    # logger.critical(f"")
    if just_ids: return local_chunks["ids"], gr.Textbox(value = len(local_chunks["ids"]))
    else: return local_chunks


def find_collections():
    '''
    Returns all collections in the database.

    Checks to see if anything is present. If there is, it will clean up the collection name into a human readable format and return it. If there is nothing, just returns an empty list.
    '''
    hr_collections = []
    client = _get_client()
    ascii_collections = client.list_collections()

    if ascii_collections: 
        hr_collections = [human_collection(ascii_collection.name) for ascii_collection in ascii_collections]
        logger.info(f"Collections found in database: {hr_collections}")
    else:
        logger.info(f"No collections found in database")

    return hr_collections


def find_document(document: str, hr_collection: str, request: gr.Request):
    '''
    Returns the number of chunks related to the documented selected. This outputs to the log.
    '''
    # from random import randint
    set_current_user(request.username)

    # ascii_collection = _clean_collection(hr_collection)
    # client = _get_client()
    # local_collection = client.get_collection(str(ascii_collection))
    local_collection = _get_local_collection(hr_collection)
    local_chunks = local_collection.get(where = {"Title": document})

    # random_chunk = randint(0, len(local_chunks["ids"]) - 1)

    # This will return a dictionary of ['ids', 'embeddings', 'documents', 'uris', 'included', 'data', 'metadatas']

    # logger.info(f"{hr_collection} | {document} | Found {len(local_chunks['ids'])} | {local_chunks['metadatas']}")
    # logger.info(f"Random Document | {local_chunks['documents'][random_chunk]}")



# Log if the collections found are empty or not. Send length of list of titles to log.
def find_documents(hr_collection, request: gr.Request):
    '''
    Finds all available documents in a given collection. Feed in the human readable collection title.
    '''
    set_current_user(request.username)
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
            logger.warning(f"Cannot find title or source in item retrieved from collection {hr_collection} | Error type {type(e)} | {e}")
        else:
            items += 1
    logger.info(f"Successfully found {items} | collection {ascii_collection}| {hr_collection}")
    
    return list(titles)


def fill_list(input_list: list, game_system: str = "") -> list:
    '''
    Makes sure that the list is populated with at least one thing.

    the list(set(list)) is to make sure that it is a list, that everything is unique, and that it is still a list, and that it's sorted.
    '''
    return list(sorted(set([game_system] + input_list) - {""}))  # the - {""} is to remove a potential "" that gets added.
    # else: return [game_system]


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


def _extract_two_column_page(page, 
                             percent = 0.2) -> str:
    '''
    '''
    words = page.extract_words()
    if not words:
        return ""

    # this finds the column split by looking for a gap in x-position density
    x_positions = sorted(set(round(w["x0"] / 10) * 10 for w in words))

    # finds the largest gap in x positions
    page_width = page.width
    mid = page_width / 2

    # Look for a gap within % of the center
    gap_start = mid * (1.0 - percent)
    gap_end = mid * (1.0 + percent)

    gaps = []

    for i in range(len(x_positions) - 1):
        if gap_start <= x_positions[i] <= gap_end:
            gap_size = x_positions[i + 1] - x_positions[i]
            gaps.append((gap_size, x_positions[i]))

    # if there's a gap, get the largest gap found
    if gaps:
        split_x = max(gaps, key = lambda g: g[0])[1] + 5
    else:
        split_x = page_width / 2

    # seperate into columns and sort each by reading order
    left = sorted([w for w in words if w['x0'] < split_x], key = lambda w: (round(w['top'] / 5) * 5, w['x0']))
    right = sorted([w for w in words if w['x0'] >= split_x], key = lambda w: (round(w['top'] / 5) * 5, w['x0']))

    left_text = " ".join(w['text'] for w in left)
    right_text = " ".join(w['text'] for w in right)

    return f"{left_text}\n\n{right_text}".strip()



def generate_summary(hr_collection: str, title: str, embed_model: str, lang_model: str, section_size: int, 
                     batch_size = 50):
    '''
    Can be used to regenerate the summary from the chunks. Needs to filter out anything that is not a summary.

    I need to think about this because if I don't save the chunks, and just generate the summary, how do I get the chunks?
    '''
    ascii_collection = _clean_collection(hr_collection)
    local_collection = _get_local_collection(hr_collection)

    try:
        local_collection.delete(where = {"Title": title, "chunk_type": "summary"})
    except ValueError as e:
        pass   # can be caused by there being nothing in there, which is OK.
    except Exception as e:
        logger.critical(f"Error deleting summary for {title} | {hr_collection} | {type(e)} | {e}")
    
    chunks = local_collection.get(where = {"Title": title})
    chunks = [Document(page_content = doc, metadata = meta if meta else {}, id = doc_id) for doc, meta, doc_id in zip(chunks.get("documents", []), chunks.get("metadatas", []), chunks.get("ids", []))]

    # This was saved and processed as a list of Documents
    
    # logger.critical(f"{type(chunks)} | {list(chunks.keys())}")

    summary = _generate_section_summary(chunks, lang_model, section_size, hr_collection)

    try:
        _load_to_Chroma(summary, ascii_collection, embed_model, add_ids = False, title = title, batch_size = batch_size)
        logger.info(f"Successfully loaded {file.name} summary | {hr_collection}")
    except Exception as e:
        logger.error(f"Failed to load summary to Chroma | {type(e)} | {e}")
        return f" Error writing summary to database: {e}"



def _generate_section_summary(chunks: list, lang_model: str, section_size: int = 10, game_system: str = "Generic") -> list:
    '''
    Generates a summary from an LLM about the chunks.
    '''
    logger.info(f"Generating Summary | {lang_model} | Section Size: {section_size}")
    summary_docs = []
    section_index = -1
    game_system = chunks[0].metadata["game_system"]
    file_path = chunks[0].metadata["source"]
    base_id = f"{game_system}:{str(pathlib.Path(file_path).stem)}:summary"


    for i in range(0, len(chunks), section_size):
        section_index += 1
        section = chunks[i: i + section_size]
        combined_text = ""
        combined_tags = set()
        for c in section:
            # combined_text = "\n\n".join(c.page_content for c in section)
            combined_text = f"{combined_text}\n\n{c.page_content}"
            combined_tags = combined_tags | set(c.metadata["tags"]) #  combined_tags | set(c.metadata["tags"].split(","))

        try:
            response = ollama.chat(model = lang_model,
                                    messages = [{"role": "user",
                                                "content": f"""Summarize the following rulebook sections in 2-5 paragraphs. Generate tables or lists if necessary.
                                                Focus on: what rules or mechanics are covered, what a player needs to know, any key terms that should be defined. Be concise but complete.

                                                Text:
                                                {combined_text}"""}])
        except Exception as e:
            logger.critical(f"Unable to generate summary | Type: {type(e)} | {e}")
            raise
        
        summary_text = response["message"]["content"]

        first_chunk = section[0]
        
        summary_doc = Document(page_content = summary_text,
                               metadata = {
                                           "embedding_used": first_chunk.metadata.get("embedding_used", "?"),  # if none, yield a ? so the user knowns that information is lost.
                                           "game_system": game_system,
                                           "tags": fill_list(list(sorted(combined_tags))), # ",".join(sorted(combined_tags)),
                                           "chunk_type": "summary",
                                           "source": file_path,
                                           "source_pages": f"{section[0].metadata.get('page', '?')}-{section[-1].metadata.get('page', '?')}",
                                           "original_chunk_count": len(section),
                                           "id": f"{base_id}:{first_chunk.metadata.get('page', '?')}-{section[-1].metadata.get('page', '?')}:{section_index}",
                                           "Title": f"{first_chunk.metadata.get("Title", game_system)}",  # this will be handled later, if None, pull from the source file
                                           "Pages": f"{first_chunk.metadata.get('page', '?')} - {section[-1].metadata.get('page', '?')}"
                                           })
        
        # logger.warning(f"{fill_list(list(sorted(combined_tags)))}")
        summary_docs.append(summary_doc)
        logger.info(f"Generated summary | {first_chunk.metadata.get('page', '?')} - {section[-1].metadata.get('page', '?')} | {base_id}:{section_index}")

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
    # if _chroma_client is None: _chroma_client = chromadb.PersistentClient(path = str(chroma_database_dir))
    if _chroma_client is None: _chroma_client = chromadb.HttpClient(host = CHROMA_HOST, port = CHROMA_PORT)
    return _chroma_client


def _get_local_collection(hr_collection):
    '''
    '''
    ascii_collection = _clean_collection(hr_collection)
    client = _get_client()
    local_collection = client.get_collection(str(ascii_collection))

    return local_collection


@lru_cache(maxsize=4)
def _get_embeddings(embed_model, search_doc = False):
    '''
    Currently this is hard coded to only use the qwen3 model embeddings.

    Part of RAG Input
    '''
    if search_doc:
        return OllamaEmbeddings(model = embed_model, model_kwargs = {"prompt": "search_document:"}, base_url = OLLAMA_HOST)
    return OllamaEmbeddings(model = embed_model, base_url = OLLAMA_HOST)


def _get_hybrid_retriever(hr_collection: str, embed_model: str, 
                          k: int = 10, alpha: float = 0.5, tags: tuple[str] | None = None):
    '''
    Makes _hybrid_search a RunnableLambda
    '''
    def retrieve(query: str) -> list[Document]:
        return _hybrid_search(query, hr_collection, embed_model, k = k, alpha = alpha, tags = tags)
    
    return RunnableLambda(retrieve)


def get_metadata(hr_collection: str, title: str, request: gr.Request):
    '''
    Gets the metadata for a given document.
    '''
    set_current_user(request.username)
    ascii_collection = _clean_collection(hr_collection)
    client = _get_client()
    collection = client.get_collection(name = ascii_collection)

    logger.info(f"Getting metadata tags for {title} | {hr_collection} | {collection}")

    results = collection.get(where = {"Title": title})
    if len(results["metadatas"]) < 1:
        return ["Document not found"], gr.Textbox(value = "Document not found")
    
    # print(results["metadatas"], list(results.keys()))
    embedding_used: str = results["metadatas"][0].get("embedding_used", "?")
    tags = set()

    for result_tags in results["metadatas"]:
        # local_tags = result_tags.split(",")
        try:
            tags.update(result_tags["tags"]) # tags.update(result_tags["tags"].split(","))
        except Exception as e:
            logger.error(f"Error tying to pull metadatas | {title} | {hr_collection} | {ascii_collection} | {type(e)} | {e}")
            continue

    return list(tags), gr.Textbox(value = embedding_used)


def _get_quarantine_collection(hr_quarantine: str):
    '''
    Gets the quarantine version of the collection, for pushing the low scoring collections to their own area to be healed later.
    '''
    client = _get_client()
    ascii_quarantine = _clean_collection(hr_quarantine)
    # quarantine_name = _quarantine_collection_name(ascii_collection)
    return client.get_or_create_collection(name = ascii_quarantine)


@lru_cache(maxsize = 8)
def _get_retriever(hr_collection: str, embed_model: str, 
                   k: int = 10, tags: tuple[str, ...] | None = None, score_threshold: float | None = None,  # score is 1 - cosine similarity, so a lower number here is a higher threshold
                   search_query: bool = False, *args, **kwargs):
    '''
    Returns a retriever for the given collection name.

    Part of RAG Query
    '''
    # if isinstance(tags, tuple): tags = list(tags)
    if not hr_collection: hr_collection = "Generic"
    ascii_collection = _clean_collection(hr_collection)
    QEM = _get_query_embeddings(embed_model, search_query)
    # db = Chroma(persist_directory = str(chroma_database_dir), embedding_function = QEM, collection_name = ascii_collection)
    db = Chroma(client=_get_client(), embedding_function = QEM, collection_name = ascii_collection)

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
    
    # logger.info(f"k = {k} | score threshold = {score_threshold} (1 - cosine similarity is used)")
    logger.critical(f"Getting DB as a retriever | {hr_collection} | {search_kwargs} | {score_threshold}")

    if score_threshold is not None:
        return db.as_retriever(search_type = "similarity_score_threshold",
                               search_kwargs = {**search_kwargs, "score_threshold": score_threshold})
    
    return db.as_retriever(search_type = "similarity", search_kwargs = search_kwargs)
    
    # return db.as_retriever(search_type = "similarity", search_kwargs = {"k": k, "filter": {"game_system": hr_collection}})


def _get_summary_retriever(hr_collection: str, embed_model: str, 
                           k: int = 5, tags: list[str] | None = None, score_threshold: float | None = None):
    '''
    Limits its retriever to only the summary chunks.
    '''
    ascii_collection = _clean_collection(hr_collection)

    # db = Chroma(persist_directory=str(chroma_database_dir), embedding_function=_get_query_embeddings(embed_model), collection_name=ascii_collection)
    db = Chroma(client=_get_client(), embedding_function=_get_query_embeddings(embed_model), collection_name=ascii_collection)

    filters = [{"game_system": hr_collection}, {"chunk_type": "summary"}]

    if tags:
        tag_filters = [{"tags": {"$contains": tag}} for tag in tags]
        filters.append({"$or": tag_filters} if len(tag_filters) > 1 else tag_filters[0])
    
    search_kwargs = {"k": k, "filter": {"$and": filters}}

    logger.critical(f"Getting DB as a retriever | {hr_collection} | {search_kwargs} | {score_threshold}")

    if score_threshold is not None:
        return db.as_retriever(search_type = "similarity_score_threshold", search_kwargs = {**search_kwargs, "score_threshold": score_threshold})

    return db.as_retriever(search_type = "similarity", search_kwargs = search_kwargs)

    
def _get_query_embeddings(embeddings: str, search_query: bool = False):
    '''
    This allows for the use of the search query prefix in nomic-embed-text. The others, mixed bread and snowflake artic, will come eventually.
    '''
    if search_query:
        return OllamaEmbeddings(model = embeddings, model_kwargs = {"prompt": "search_query:"}, base_url = OLLAMA_HOST)
    return OllamaEmbeddings(model = embeddings, base_url = OLLAMA_HOST)


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



def heal_chunk(id: str, text: str, tags: list[str], hrq_collection: str, embed_model: str,
               request: gr.Request):
    '''
    Heals a chunk and moves it to the main database.

    Creates a new document from the existing data and new data of the chunk, saves that chunk to the main collection, then deletes the quarantined data.
    
    This effectively overwrites the chunk and forces it into the healed area. This means you could just click "heal chunk" and it will be healed.
    '''
    # get original metadatas
    hr_collection = hrq_collection.replace("_quarantine", "")  # removes the quarantine part
    
    db = Chroma(client = _get_client(), embedding_function=_get_embeddings(embed_model), collection_name = _clean_collection(hr_collection))
    dbq = Chroma(client = _get_client(), embedding_function=_get_embeddings(embed_model), collection_name = _clean_collection(hrq_collection))
    
    local_chunk = dbq.get(ids = [id])

    new_scores: dict = _score_chunk_quality(local_chunk["metadata"], text)  # returns {scores: score}
    metadatas: dict = local_chunk["metadata"][0]
    metadatas["tags"] = tags
    metadatas = {**metadatas, **new_scores}

    healed_doc = Document(page_content = text, metadata = metadatas, id = id)

    db.add_documents(documents = [healed_doc], ids = id)

    delete_chunk(id, hrq_collection, request)  # this deletes the quarantined chunk



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


def _hybrid_search(question: str, hr_collection: str, embed_model: str, 
                  k: int = 10, alpha: float = 0.5, tags: tuple[str] | None = None) -> list:
    '''
    '''
    ascii_collection = _clean_collection(hr_collection)
    client = _get_client()
    raw_collection = client.get_collection(str(ascii_collection))

    # logger.warning(f"Raw Collection | {type(raw_collection)}")
    # logger.warning(f"{raw_collection['metadatas']}")

    where_filter = {"game_system": hr_collection}
    if tags:
        # tag_filters = [{"tags": {"$contains": ",".join(sorted(tags))}}]
        tag_filters = [{"tags": {"$contains": tag}} for tag in tags]
        # where_filter = {"$and": [where_filter, {"$or": tag_filters}]}
        if len(tag_filters) == 1:
            where_filter = {"$and": [where_filter, tag_filters[0]]}
        else:
            where_filter = {"$and": [where_filter, {"$or": tag_filters}]}

    logger.warning(f"Using the following where filter | {where_filter}")

    all_data = raw_collection.get(where = where_filter, include = ["documents", "embeddings", "metadatas"])
    all_docs = all_data["documents"]
    all_ids = all_data["ids"]
    all_metas = all_data["metadatas"]

    if not all_docs:
        logger.critical(f"Hybrid search failed | No Documents found | {hr_collection} | {tags}")
        return []

    tokenized = [doc.lower().split() for doc in all_docs]
    bm25 = BM25Okapi(tokenized)
    bm25_scores = bm25.get_scores(question.lower().split())

    # db = Chroma(persist_directory=str(chroma_database_dir), embedding_function=_get_embeddings(embed_model), collection_name=ascii_collection)
    db = Chroma(client=_get_client(), embedding_function=_get_embeddings(embed_model), collection_name=ascii_collection)
    semantic_results = db.similarity_search_with_score(question, k = len(all_docs), filter = where_filter)

    semantic_score_map = {result.metadata.get("id", ""): score for result, score in semantic_results}

    max_possible_distance = 2.0

    semantic_raw = [semantic_score_map.get(doc_id, max_possible_distance) for doc_id in all_ids]

    bm25_normalized = _normalize_scores(bm25_scores, flip = False)
    semantic_normalized = _normalize_scores(semantic_raw, flip = True)

    combined_scores = [alpha * sem + (1 - alpha) * bm25 for sem, bm25 in zip(semantic_normalized, bm25_normalized)]

    ranked_indices = sorted(range(len(combined_scores)), key = lambda i: combined_scores[i], reverse = True)
    
    logger.info(f"Hybrid search top {min(k, len(ranked_indices))} results:")
    for rank, idx in enumerate(ranked_indices[:k]):
        logger.info(
            f"  Rank {rank+1} | Combined: {combined_scores[idx]:.3f} | "
            f"Semantic: {semantic_normalized[idx]:.3f} | "
            f"BM25: {bm25_normalized[idx]:.3f} | "
            f"Page: {all_metas[idx].get('page', '?')} | "
            f"Preview: {all_docs[idx][:60]}"
        )

    return [Document(page_content=all_docs[idx], metadata = all_metas[idx]) for idx in ranked_indices[:k]]


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
    
    for doc in document:
        original_len = len(doc.page_content)
        doc.page_content = _clean_double_chars(doc.page_content)
        if len(doc.page_content) < original_len * 0.7: logger.warning(f"Significant de-duplication on page {doc.metadata.get('page', '?')} | {original_len} -> {len(doc.page_content)} chars")
    
    return document



def _load_document_column_aware(file_path: pathlib.Path) -> list:
    '''
    This checks for two columns and tries to parse if it sees an unusual position of them (think Delta Green).

    What this also does is score each page, which really should be done as a separate area.
    '''
    # import pdfplumber
    documents = []

    with pdfplumber.open(str(file_path)) as pdf:
        og_metadata = pdf.metadata
        for i, page in enumerate(pdf.pages):
            words = page.extract_words()
            if not words: continue

            x_positions = [w['x0'] for w in words]
            page_width = page.width
            mid = page_width / 2

            left_count = sum(1 for x in x_positions if x < mid * 0.85)
            right_count = sum(1 for x in x_positions if x > mid * 1.15)
            center_count = sum(1 for x in x_positions if mid * 0.85 <= x <= mid * 1.15)

            is_two_column = (left_count > 20 and right_count > 20 and center_count < (left_count + right_count) * 1.15)

            if is_two_column:
                text: str = _extract_two_column_page(page)
                extraction_method = "two_column"
            else:
                text: str = page.extract_text() or ""
                extraction_method = "standard"

            page_score = _score_page_quality(text, page.chars, page.images, page.width, page.height)
            
            if text.strip():
                documents.append(Document(page_content=text, metadata = {**og_metadata, "source": str(file_path), "page": i, "extraction_method": extraction_method, **page_score}))

            # logger.info(f"Page {i}: {extraction_method} extraction | {len(text)} chars")

    return documents



def load_documents(file, hr_collection, embed_model, lang_model, request: gr.Request, vision_model: str | None = None,
                   tags: list | None = None, chunk_size = 512, chunk_overlap = 50, chunk_batch = 50, chunk_sum = 10, 
                   save_chunks: bool = True, save_summary: bool = True,
                   *args, **kwargs):
    '''
    Loads the document from the input path, then add it to the database.

    Part of RAG Input

    if chunks -> gen chunks, save chunks
    if summary -> gen chunks, gen summary, save summary
    if both -> gen chunks, save chunks, gen summary, save summary
    '''
    set_current_user(request.username)

    logger.info(f"Loading file | {file} | {hr_collection} | Embedding Model: {embed_model} | Language Model: {lang_model} | Vision Model: {vision_model} | C Size: {chunk_size} | C Overlap: {chunk_overlap}")
    
    ascii_collection = _clean_collection(hr_collection)
    if isinstance(file, str): file = pathlib.Path(file)
    title = file.stem  # this is to ensure that there is a title being added to the metadata. Also it's better than using hr_collection because that represents the rule system
    document = None
    chunks = None
    DocLoader = None

    suffix_map = {
        ".csv": UnstructuredCSVLoader, 
        ".doc": UnstructuredWordDocumentLoader,
        ".docx": UnstructuredWordDocumentLoader, 
        ".epub": UnstructuredEPubLoader, 
        ".md": UnstructuredMarkdownLoader,
        ".pdf": PDFPlumberLoader, # this is kinda not used much anymore, because of the possibility of dual columns.
        ".txt": TextLoader, 
                  }
    
    DocLoader = suffix_map.get(file.suffix)

    if DocLoader is None:
        logger.error(f"Unsupported file type | {file.suffix}")
        return f"Error: Unsupported file type: {file.suffix}"
    
    logger.info(f"Document loader set to {DocLoader.__name__}")

    try: 
        if file.suffix == ".pdf": 
           if vision_model and vision_model != "NONE": document = _sort_pdf_pages(file, vision_model) # _load_pdf_multimodal(file, vision_model)
           else: document = _load_document_column_aware(file)
        else: document = _load_document(file, DocLoader)
    except PermissionError as e:
        logger.error(f"Failed to load document | {file} | Permission Error")
        return f"Permission Error: Please try again"
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
    
    chunks = _enrich_chunk_metadata(chunks, embed_model = embed_model, game_system = hr_collection)
    if tags:
        user_tags = set(tags) if isinstance(tags, list) else {tags}
        for chunk in chunks:
            existing = set(chunk.metadata.get("tags", "")) - {""} # set(chunk.metadata.get("tags", "").split(",")) - {""}
            chunk.metadata["manual_tags"] = list(sorted(user_tags))
            chunk.metadata["tags"] = list(sorted(existing | user_tags))
    
    logger.info(f"Created {len(chunks)} chunks from {file.name} | Enriched with automated metadata")

    if save_chunks:
        try:
            _load_to_Chroma(chunks, ascii_collection, embed_model, tags = tags or [], title = title, batch_size = chunk_batch)
            logger.info(f"Successfully loaded {file.name} | {hr_collection} | {tags}")
            # return f"Successfully added {len(chunks)} chunks | {file.name} | {hr_collection}."
        except Exception as e:
            logger.error(f"Failed to load chunks to Chroma | {type(e)} | {e}")
            return f"Error writing to database: {e}"
        
    if save_summary:
        summary = _generate_section_summary(chunks, lang_model, chunk_sum, game_system = title)

        try:
            _load_to_Chroma(summary, ascii_collection, embed_model, add_ids = False, title = title, batch_size = chunk_batch)
            logger.info(f"Successfully loaded {file.name} summary | {hr_collection}")
        except Exception as e:
            logger.error(f"Failed to load summary to Chroma | {type(e)} | {e}")
            return f" Error writing summary to database: {e}"


def _load_to_Chroma(chunks, ascii_collection, embed_model, 
                    add_ids: bool = True, batch_size = 50, game_system: str | None = None, title: str | None = None,
                    *args, **kwargs):
    '''
    Loads the documents to a Chroma DB

    Part of RAG Input
    '''
    hr_collection = human_collection(ascii_collection)
    embeddings = _get_embeddings(embed_model)
    # db = Chroma(persist_directory = str(chroma_database_dir), embedding_function = _get_embeddings(embed_model), collection_name = collection)  # get embeddings here never uses the search doc prefix. Think about turning that on at some point.
    db = Chroma(client=_get_client(), embedding_function = embeddings, collection_name = ascii_collection)  # get embeddings here never uses the search doc prefix. Think about turning that on at some point.

    quar_db = _get_quarantine_collection(f"{hr_collection}_quarantine")

    existing_items = db.get(include = [])
    existing_ids = set(existing_items["ids"])

    if add_ids: chunks = _metadata_IDs(chunks, unique_key = len(existing_ids), title = title)

    try:
        existing_quarantine = quar_db.get(include = [])
        existing_ids |= set(existing_quarantine["ids"])  # adds the quarantine collection to the stuff, to make sure that nothing is re-ingested by accident.
    except Exception as e:
        pass  # I can't believe Claude would write a bare except. I mean it's probably nothing, but still.

    # new_chunks = [c for c in chunks if c.metadata["id"] not in existing_ids]
    new_chunks = []
    quar_chunks = []

    for chunk in chunks:
        if chunk.metadata["id"] in existing_ids: continue
        else:
            chunk.metadata |= _score_chunk_quality(chunk.metadata, chunk.page_content)
            chunk.metadata["Title"] = chunk.metadata.get("Title", title)  # Just reassigning it, because for some reason the title isn't always there. I should pull apart a bunch of PDFs to see if the title is always there or not.

        if chunk.metadata.get("quality_pass", True): new_chunks.append(chunk)
        else: quar_chunks.append(chunk)

    if new_chunks:
        logger.info(f"Adding {len(new_chunks)} new chunks | Collection: {hr_collection} | Batches: {batch_size} ")

        for i in range(0, len(new_chunks), batch_size):
            batch = new_chunks[i: i + batch_size]
            batch_ids = [c.metadata["id"] for c in batch]
            try:
                db.add_documents(batch, ids = batch_ids)
                logger.info(f"Batch {i // batch_size + 1}/{-(-len(new_chunks)//batch_size)} complete | {min(i + batch_size, len(new_chunks))}/{len(new_chunks)} chunks added")
            except Exception as e:
                logger.error(f"Failed on batch {i//batch_size + 1} | {type(e)} | {e}")
                raise

    else:
        logger.warning(f"No new chunks to add to {hr_collection}")

    if quar_chunks:
        # dimensions = embeddings.dimensions
        dimension_vector = OllamaEmbeddings(model = embed_model)
        vector = dimension_vector.embed_query("dimension check")
        dimensions = len(vector)  # this creates a dummy vector to figure out what the vector size is becuase embeddings.dimensions didn't work, came back with None

        logger.warning(f"Adding {len(quar_chunks)} new chunks | Collection: {hr_collection}_quarantine | Batches: {batch_size}")

        q_ids = [c.metadata["id"] for c in quar_chunks]
        q_docs = [c.page_content for c in quar_chunks]
        q_metas = [c.metadata for c in quar_chunks]

        try:
            quar_db.add(
                        ids = q_ids,
                        documents = q_docs,
                        metadatas = q_metas,
                        embeddings = [[0.0] * dimensions] * len(quar_chunks)
                        )
        except Exception as e:
            logger.error(f"Failed to quarantine files | {type(e)} | {e}")
            raise
        else:
            logger.warning(f"Finished adding quarantined chunks | {hr_collection}_quarantine")
    



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


def _metadata_IDs(chunks, title: str | None = None, unique_key = 0):
    '''
    Assigns a new metadata ID to the item. The metadata tag is: source document: page: chunk index. The chunk index for each document goes from [0, max chunks].
    
    Part of RAG Input
    '''
    last_page_id = None
    current_chunk_index = 0
    # title = title or "?"
    title = chunks[0].metadata.get("Title", title) or f"Doc_{unique_key}"
    base_page_id = f"{chunks[0].metadata.get('game_system', f'Generic_{unique_key}')}:{title}:chunk"

    # there could be something here, like checking to see if the metadata was properly added, but I'm not sure. I might be logging things to log them.
    for chunk in chunks:
        # source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        current_page_id = f"{base_page_id}:{page}"

        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0

        chunk_id = f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id

        chunk.metadata["id"] = chunk_id

    logger.warning(f"{chunks[0].metadata}")

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


def _normalize_scores(scores: list[float], flip: bool = False) -> list[float]:
    '''
    Min-max normalizes the scores to [0, 1], the traditional normalization.

    Flip = True inverts the results, useful for scores where lower = higher similarity
    '''
    # if not scores: return []
    if len(scores) == 0: return []

    min_s = min(scores)
    max_s = max(scores)

    if max_s == min_s:
        return [1.0 if not flip else 0.0] * len(scores)
    
    normalized = [(s - min_s) / (max_s - min_s) for s in scores]

    if flip:
        normalized = [1.0 - n for n in normalized]

    return normalized


def _quarantine_collection_name(ascii_collection: str) -> str:
    '''
    Adds __quarantine to the end of the collection name, making it something that cannot be queried with the rest of the documents
    '''
    name = f"{ascii_collection}__quarantine"
    return name[:512]


def query_rag_routed(message: str, history: list, lang_model: str, embed_model: str, 
                     collection: str | None = None, tags: list[str] | None = None, k: int = 10, score_threshold: float | None = None,
                     *args, **kwargs):
    '''
    Routes the question to the correct response, or at least the response that matches the question.
    '''
    if isinstance(tags, str): 
        logger.warning(f"Tags had a string input | {tags} | Setting to NONE")
        tags = None

    # from __tech_fn import load_tags
    # available_tags = load_tags()
    available_tags = enrich_keywords  # because I'm lazy and don't want to change the variable right now.
    rule_systems = find_collections()

    # llm = ChatOllama(model = lang_model)
    llm = ChatOllama(model = lang_model, base_url = OLLAMA_HOST)

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
    
    # The forcing of the float here is to make sure that the number going in is a float.
    if len(collections_to_search) == 1:
        retriever = _get_hybrid_retriever(collections_to_search[0], embed_model, k = k, tags = tags)
    else:
        def multi_retriever(query: str) -> list[Document]:
            '''
            Probably could be a stand alone function rather than an embedded one.
            '''
            seen = set()
            merged = []
            for col in collections_to_search:
                results = _hybrid_search(query, col, embed_model, k = max(3, k // len(collections_to_search)), tags = tags)
                for doc in results:
                    if doc.page_content not in seen:
                        seen.add(doc.page_content)
                        merged.append(doc)
            return merged[:k]
        
        retriever = RunnableLambda(multi_retriever)
    
    logger.info(f"Generating query | {lang_model} | {embed_model} | {type(retriever)}")
    
    yield from _rag_response(message, history, lang_model, retriever)


def _rag_response(message, history, lang_model, retriever):
    '''
    Shared RAG generation logic. Separated so both query_rag and query_rag_routed can use it without duplication.

    However query_rag is not going to be used in the future, so this is instead a nice way to apply logging logic and find errors.
    '''
    # llm = ChatOllama(model = lang_model)
    llm = ChatOllama(model = lang_model, base_url = OLLAMA_HOST)

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

    response = ""
    try:
        for chunk in chain.stream({"question": message, "history": lc_history}):
            response += chunk
            yield response
    except chromadb.errors.InvalidArgumentError as e:
        logger.critical(f"Unable to return message | check embedding model being used | {e}")
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



def _render_page_as_base64(page: PageClass, resolution: int = 300) -> str:
    '''
    Renders a PDF page as base64 by first turning it into a png, which then is converted to a string of base64.
    '''
    # with pdfplumber.open(pdf_path) as pdf:
        # page = pdf.pages[page_num]
    pdf_image = page.to_image(resolution = resolution)

    buffer = io.BytesIO()
    pdf_image.original.save(buffer, format="PNG")
    buffer.seek(0)

    return base64.b64encode(buffer.read()).decode()


def _score_chunk_quality(chunk_metadata: dict, text: str) -> dict:
    '''
    Scores a chunk, which will determine if it goes into quarantine or not.
    '''
    words = text.split()
    word_count = len(words)

    # checks the double character ratio
    doubled = sum(1 for i in range(len(text) - 1) if text[i] == text[i + 1] and text[i].isalpha())
    doubled_ratio = doubled / max(len(text), 1)

    # word length distribution
    ave_word_len = sum(len(w) for w in words) / max(word_count, 1)

    # very short ave word len suggests scattered single characters
    # very long suggests words are being merged without spaces
    word_len_suspicious = ave_word_len < 2.5 or ave_word_len > 12

    angled_ratio = chunk_metadata.get("angled_ratio", 0.0)

    scores = {
        "angled_ratio": angled_ratio,
        "doubled_ratio": doubled_ratio,
        "word_count": word_count,
        "ave_word_len": round(ave_word_len, 2),
        "word_len_suspicious": word_len_suspicious,
        "text_length": len(text),
        "has_images_only": chunk_metadata.get("is_sparse", False) and chunk_metadata.get("has_images", False),
        "extraction_method": chunk_metadata.get("extraction_method", "unknown")
    }

    # pass/fail
    scores["quality_pass"] = (angled_ratio < QUALITY_THRESHOLDS["angled_ratio"]
                              and doubled_ratio < QUALITY_THRESHOLDS["max_doubled_ratio"]
                              and word_count >= QUALITY_THRESHOLDS["min_word_count"]
                              and len(text) >= QUALITY_THRESHOLDS["min_text_length"]
                              and not word_len_suspicious
                              and not scores["has_images_only"])
    
    scores["quality_score"] = round(
        1.0
        - (angled_ratio * 0.4)
        - (doubled_ratio * 0.3)
        - (0.2 if word_len_suspicious else 0.0)
        - (0.1 if scores["has_images_only"] else 0.0),
        3
    )

    return scores



def _score_page_quality(text: str, chars, page_images, page_width, page_height) -> dict:
    '''
    Scores the page on a few different metrics:

    Is it angled? This will require a vision model
    Does is have a lot of images? This may also require a vision model or can be skipped.
    What is the text density? If low then it might be the captioning of an image.
    Is it sparse? This could be related to the previous metric, or it could be a chapter title, etc.
    '''
    # chars = page.chars
    total_chars = len(chars)
    words = text.split()

    angled_ratio = sum(1 for c in chars if abs(c.get('matrix', (0,0,0,0,0,0))[1]) >= 0.1) / max(total_chars, 1)
    ave_word_len = (sum(len(w) for w in words) / len(words)) if words else 0
    # has_images = len(page.images) > 0
    # text_density = len(text) / max(page.width * page.height, 1)
    has_images = len(page_images) > 0
    text_density = len(text) / max(page_width * page_height, 1)
    is_sparse = len(text) < 100

    return{
        "angled_ratio": angled_ratio,
        "ave_word_len": ave_word_len,
        "has_images": has_images,
        "text_density": text_density,
        "is_sparse": is_sparse,
        "is_suspicious": ((angled_ratio > 0.2) or (ave_word_len < 2.5) or (ave_word_len > 12) or (len(text) < 100 and has_images))
    }


def _sort_pdf_pages(file_path: pathlib.Path | str, vis_model: str) -> list:
    '''
    This is a handler for sorting the pdf pages to be either read with the two column approach or with the multimodal approach.
    
    It will first score the page, then it will sort the page based on the score. (Do the scores then need to be updated? Not sure)

    Good scores go straight to the two column aware function. Bad scores go to the multimodal extraction after getting the two column treatment.
    '''
    documents = []
    with pdfplumber.open(str(file_path)) as pdf:
        og_metadata = pdf.metadata
        total_pages = len(pdf.pages)

        for p, page in enumerate(pdf.pages):
            words = page.extract_words()
            if not words:
                logger.warning(f"No words extracted, attempting to use {vis_model} for extraction | {p} / {total_pages}")
                try:
                    text = _extract_page_multimodal(page, vis_model)
                    extraction_method = f"Multimodal: {vis_model}: no words"
                except Exception as e:
                    logger.error(f"Unable to extract words or text from page {p}")
                    continue 
            else:
                if _two_columns(words, page.width):
                    text: str = _extract_two_column_page(page)
                    extraction_method = "two_column"
                else:
                    text: str = page.extract_text() or ""
                    extraction_method = "standard"

            page_score = _score_page_quality(text, page.chars, page.images, page.width, page.height)

            if page_score["is_suspicious"]:
                logger.info(f"Page {p} is suspicious | Angle Ratio: {page_score['angled_ratio']} | Average Word Length: {page_score['ave_word_len']} | Has Images: {page_score['has_images']} | Text Density: {page_score['text_density']} | Is Sparse: {page_score['is_sparse']}")

                try:
                    text = _extract_page_multimodal(page, vis_model)
                    extraction_method = f"Multimodal: {vis_model}"
                except Exception as e:
                    logger.error(f"Vision Extraction failed on page {p} | {(type(e))} | {e}")
                    extraction_method = f"{extraction_method}: Multimodal Failed"

            if text.strip():
                documents.append(Document(
                    page_content=text,
                    metadata = {**og_metadata,
                                "source": str(file_path),
                                "page": p,
                                "extraction_method": extraction_method,
                                **page_score}))

            if p % 10 == 0:
                logger.info(f"Progress: {p} / {total_pages} pages processed")
        
        vision_count = sum(1 for d in documents if d.metadata.get("extraction_method") == f"Multimodal: {vis_model}")

        logger.info(f"Extraction complete | Visual Extraction extraction: {vision_count} | Text extraction: {len(documents) - vision_count}")

    return documents



def _two_columns(words, page_width):
    '''
    Determines if the pdf has two columns or not.
    '''
    x_positions = [w['x0'] for w in words]
    mid = page_width / 2

    left_count = sum(1 for x in x_positions if x < mid * 0.85)
    right_count = sum(1 for x in x_positions if x > mid * 1.15)
    center_count = sum(1 for x in x_positions if mid * 0.85 <= x <= mid * 1.15)

    is_two_column = (left_count > 20 and right_count > 20 and center_count < (left_count + right_count) * 1.15)

    return is_two_column



def update_chunk(id, metadatas: list[str], hr_collection, embed_model,
                 text: str | None = None):
    '''
    Updates a *specific* document.

    Still trying to figure this out
    '''
    metadatas: dict = {"tags": metadatas}
    db = Chroma(client=_get_client(), embedding_function = _get_embeddings(embed_model), collection_name = _clean_collection(hr_collection))
    local_chunk = db.get(ids = [id])
    # print(local_chunk["documents"])  # they return an empty list?
    # print(local_chunk["metadatas"])

    if text: updated_doc = Document(page_content = text, metadata = {**local_chunk["metadatas"][0], **metadatas}, id = id)

    else: updated_doc = Document(page_content = local_chunk["documents"][0], metadata = {**local_chunk["metadatas"][0], **metadatas}, id = id)
    
    db.update_document(document_id=id, document=updated_doc)
    
    logger.info(f"Document {id} updated | {hr_collection}")



def update_metadata(hr_collection: str, title: str, new_tags: list, request: gr.Request):
    '''
    Updates the metadata with new tags.

    This updates individual chunks, so it is theoretically possible to target specific chunks for different metadata.
    '''
    set_current_user(request.username)
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

    for meta_chunk in local_result["metadatas"]:
        old_tags = set(meta_chunk.get("tags", "")) - {""} # set(meta.get("tags", "").split(",")) - {""} # this strips the empty string. Useful for if "tags" is empty.
        merged_tags: set = old_tags | new_tags
        updated_meta = {**meta_chunk, "tags": fill_list(list(sorted(merged_tags))) }# ",".join(sorted(merged_tags))}  # unpack, sort the list, and join the whole thing into a string
        updated_metadatas.append(updated_meta)

    local_collection.update(ids = local_result["ids"], metadatas =  updated_metadatas)

    logger.info(f"Finished updating metadata | {title} | {hr_collection} | {updated_meta}")




logger.info(f"Finished reading RAG Pipeline file @ (time to be implemented)")