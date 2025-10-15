from langchain_core.documents import Document

def dicts_to_documents(chunks):
    """
    Convert a list of dicts with 'page_content' and 'metadata' to LangChain Document objects.
    """
    return [Document(page_content=chunk['page_content'], metadata=chunk.get('metadata', {})) for chunk in chunks]

from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
import re
from langchain_community.document_loaders import UnstructuredFileLoader
import docx2txt
import pandas as pd
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS

def load_documents(data_dir):
    """
    Load documents from PDF, TXT, DOCX, and CSV files in the given directory.
    Returns a list of LangChain Document objects.
    """
    documents = []
    # PDF
    pdf_loader = DirectoryLoader(data_dir, glob="*.pdf", loader_cls=PyPDFLoader)
    documents.extend(pdf_loader.load())
    # TXT
    txt_loader = DirectoryLoader(data_dir, glob="*.txt", loader_cls=UnstructuredFileLoader)
    documents.extend(txt_loader.load())
    # DOCX
    import os
    for file in os.listdir(data_dir):
        if file.endswith(".docx"):
            text = docx2txt.process(os.path.join(data_dir, file))
            documents.append({'page_content': text, 'metadata': {'source': file}})
    # CSV
    for file in os.listdir(data_dir):
        if file.endswith(".csv"):
            df = pd.read_csv(os.path.join(data_dir, file))
            text = df.to_string()
            documents.append({'page_content': text, 'metadata': {'source': file}})
    return documents

def chunk_by_headings(documents, headings=None):
    """
    Split documents by medical headings (e.g., Symptoms, Diagnosis, Treatment).
    Falls back to RecursiveCharacterTextSplitter if headings not found.
    """
    if headings is None:
        headings = [r"Symptoms:", r"Diagnosis:", r"Treatment:", r"Definition:", r"Prognosis:"]
    pattern = re.compile(r"|".join(headings), re.IGNORECASE)
    chunks = []
    # ...rest of your logic...
