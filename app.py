from flask import Flask, render_template, jsonify, request
from src.helper import download_hugging_face_embeddings, hybrid_search, load_documents, chunk_by_headings, use_faiss_vectorstore
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Pinecone, FAISS
from langchain_openai import OpenAI, ChatOpenAI
from langchain_community.llms import HuggingFaceHub
from langchain.chains.summarize import load_summarize_chain
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from src.prompt import *
import os

app = Flask("MedAssist Pro")

# Load environment variables from .env file
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Allow running without Pinecone by using FAISS offline mode
use_faiss = not bool(PINECONE_API_KEY)

if PINECONE_API_KEY:
    os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
if OPENAI_API_KEY:
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY


# Model selection logic
MODEL_CHOICES = {
    "gpt-3.5-turbo-instruct": lambda: ChatOpenAI(model="gpt-3.5-turbo-instruct"),
    "biogpt": lambda: HuggingFaceHub(repo_id="microsoft/biogpt"),
    "mistral": lambda: ChatOpenAI(model="mistral-small"),
}

def get_llm(model_choice):
    return MODEL_CHOICES.get(model_choice, MODEL_CHOICES["gpt-3.5-turbo-instruct"])()

print("[INFO] Downloading HuggingFace embeddings...")
embeddings = download_hugging_face_embeddings()
print("[INFO] Embeddings loaded.")

# Use FAISS if Pinecone API key is not available
if use_faiss:
    print("[INFO] Using FAISS for local vector search (offline mode).")
    docs = load_documents('Data/')
    print(f"[INFO] Loaded {len(docs)} documents from Data/")
    chunks = chunk_by_headings(docs)
    print(f"[INFO] Chunked into {len(chunks)} sections.")
    from src.helper import dicts_to_documents
    doc_objs = dicts_to_documents(chunks)
    print(f"[INFO] Converted chunks to {len(doc_objs)} Document objects.")
    docsearch = use_faiss_vectorstore(doc_objs, embeddings)
    print("[INFO] FAISS vector store created.")
else:
    index_name = "medassistpro"
    print("[INFO] Using Pinecone vector store.")
    docsearch = Pinecone.from_existing_index(index_name=index_name, embedding=embeddings)
    print("[INFO] Pinecone vector store loaded.")

retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k":3})

# Session memory for chat - using simple list to track conversation history
chat_history = []

def get_rag_chain(model_choice):
    llm = get_llm(model_choice)
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(retriever, question_answer_chain)


@app.route("/")
def index():
    return render_template('chat.html')



# Chat endpoint with model selection, hybrid search, and source display
@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    model_choice = request.form.get("model", "gpt-3.5-turbo-instruct")
    retriever_type = request.form.get("retriever", "pinecone")
    print(f"User input: {msg} | Model: {model_choice} | Retriever: {retriever_type}")
    # Custom greeting response
    if msg.strip().lower() == "hello":
        return "Hello, How can I assist you?"
    # Hybrid search
    docs = hybrid_search(msg, retriever_type=retriever_type, pinecone_index=docsearch)
    rag_chain = get_rag_chain(model_choice)
    response = rag_chain.invoke({"input": msg, "chat_history": chat_history})
    answer = response.get("answer", "")
    sources = [doc.metadata.get("source", "Unknown") for doc in docs]
    # Save to chat history
    chat_history.append({"role": "user", "content": msg})
    chat_history.append({"role": "assistant", "content": answer})
    # Keep only last 10 exchanges
    if len(chat_history) > 20:
        chat_history[:] = chat_history[-20:]
    source_html = "<br>Source: " + ", ".join(sources) if sources else ""
    return str(answer) + source_html

# Summarization endpoint
@app.route("/summarize", methods=["POST"])
def summarize():
    file_path = request.form["file"]
    model_choice = request.form.get("model", "gpt-3.5-turbo-instruct")
    llm = get_llm(model_choice)
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    chain = load_summarize_chain(llm, chain_type="stuff")
    summary = chain.run(docs)
    return jsonify({"summary": summary})




if __name__ == '__main__':
    print("[INFO] Starting Flask server on http://0.0.0.0:8080 ...")
    app.run(host="0.0.0.0", port=8080, debug=True)
