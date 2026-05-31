# For checking that every dependency is installed correctly.
print("This will check to make sure all needed Python libraries are present by trying to import them. Make sure you are on the proper Python environment when running this script. If they are not, please find install them via Anaconda or pip. Importing libraries now...")

try:
    import chromadb
except Exception as e:
    print("Cannot find Chroma DB installed, please install either via Anaconda or pip")
    print(f"Error type {e}\nError: {e}")

try:
    import gradio as gr
except Exception as e:
    print("Cannot find Gradio installed, please install either via Anaconda or pip")
    print(f"Error type {e}\nError: {e}")

try:
    import langchain
except Exception as e:
    print("Cannot find LangChain installed, please install either via Anaconda or pip")
    print(f"Error type {e}\nError: {e}")

try:
    from langchain_chroma import Chroma
except Exception as e:
    print("Cannot find LangChain-Chroma installed, please install either via Anaconda or pip")
    print(f"Error type {e}\nError: {e}")

try:
    from langchain_community.document_loaders import PDFPlumberLoader
except Exception as e:
    print("Cannot find LangChain-Community installed, please install either via Anaconda or pip")
    print(f"Error type {e}\nError: {e}")

try:
    import langchain_ollama
except Exception as e:
    print("Cannot find LangChain-Ollama installed, please install either via Anaconda or pip")
    print(f"Error type {e}\nError: {e}")

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except Exception as e:
    print("Cannot find LangChain installed, please install either via Anaconda or pip")
    print(f"Error type {e}\nError: {e}")

try:
    import os
except Exception as e:
    print("Cannot find OS installed, please install either via Anaconda or pip")
    print(f"Error type {e}\nError: {e}")

try:
    import ollama
except Exception as e:
    print("Cannot find Ollama installed, please install either via Anaconda or pip. Additionally, check that it is running before starting Technomancer")
    print(f"Error type {e}\nError: {e}")

try:
    import pandas
except Exception as e:
    print("Cannot find Pandas installed, please install either via Anaconda or pip")
    print(f"Error type {e}\nError: {e}")

try:
    import pathlib
except Exception as e:
    print("Cannot find Pathlib installed, please install either via Anaconda or pip")
    print(f"Error type {e}\nError: {e}")

try:
    import pdfplumber
except Exception as e:
    print("Cannot find PDF Plumber installed, please install either via Anaconda or pip")
    print(f"Error type {e}\nError: {e}")

try:
    import re
except Exception as e:
    print("Cannot find Re (regex) installed, please install either via Anaconda or pip")
    print(f"Error type {e}\nError: {e}")

try:
    import socket
except Exception as e:
    print("Cannot find Socket installed, please install either via Anaconda or pip")
    print(f"Error type {e}\nError: {e}")

try:
    import yaml
except Exception as e:
    print("Cannot find Yaml installed, please install either via Anaconda or pip")
    print(f"Error type {e}\nError: {e}")
