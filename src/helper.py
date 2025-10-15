from langchain_core.documents import Document
def dicts_to_documents(chunks):
    """
    Convert a list of dicts with 'page_content' and 'metadata' to LangChain Document objects.
    """
    return [Document(page_content=chunk['page_content'], metadata=chunk.get('metadata', {})) for chunk in chunks]
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface.embeddings import HuggingFaceEmbeddings


#Extract Data From the PDF File
"""
MedAssist Pro - Professional Virtual Health Assistant
Helper functions for document loading, text splitting, and embeddings.
"""


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
    for doc in documents:
        content = doc['page_content'] if isinstance(doc, dict) else doc.page_content
        metadata = doc.get('metadata', {}) if isinstance(doc, dict) else doc.metadata
        splits = pattern.split(content)
        if len(splits) > 1:
            for split in splits:
                if split.strip():
                    chunks.append({'page_content': split.strip(), 'metadata': metadata})
        else:
            # fallback
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
            chunks.extend([{'page_content': chunk.page_content, 'metadata': chunk.metadata} for chunk in text_splitter.split_documents([doc])])
    return chunks

def hybrid_search(query, retriever_type="pinecone", docs=None, embeddings=None, pinecone_index=None):
    """
    Perform hybrid search using Pinecone, BM25, or FAISS.
    """
    if retriever_type == "pinecone" and pinecone_index:
        return pinecone_index.similarity_search(query, k=3)
    elif retriever_type == "bm25" and docs:
        bm25 = BM25Retriever.from_documents(docs)
        return bm25.get_relevant_documents(query)
    elif retriever_type == "faiss" and docs and embeddings:
        faiss_store = FAISS.from_documents(docs, embeddings)
        return faiss_store.similarity_search(query, k=3)
    else:
        raise ValueError("Invalid retriever_type or missing arguments.")

def use_faiss_vectorstore(docs, embeddings):
    """
    Create a FAISS vector store from documents and embeddings.
    """
    return FAISS.from_documents(docs, embeddings)



#Split the Data into Text Chunks
# Split the data into text chunks
def text_split(extracted_data):
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
    text_chunks=text_splitter.split_documents(extracted_data)
    return text_chunks



#Download the Embeddings from HuggingFace 
# Download professional embeddings from HuggingFace
def download_hugging_face_embeddings():
    embeddings=HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')  #this model return 384 dimensions
    return embeddings