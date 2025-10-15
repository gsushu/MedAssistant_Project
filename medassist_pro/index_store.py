from medassist_pro.helper import load_documents, chunk_by_headings, download_hugging_face_embeddings, use_faiss_vectorstore
from pinecone.grpc import PineconeGRPC as Pineconegrpc
from pinecone import ServerlessSpec
from langchain.vectorstores import Pinecone
from dotenv import load_dotenv
import os

load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY

embeddings = download_hugging_face_embeddings()
docs = load_documents('data/')
chunks = chunk_by_headings(docs)

use_faiss = not bool(PINECONE_API_KEY)

if use_faiss:
    print("[INFO] Using FAISS for local vector search (offline mode).")
    vectorstore = use_faiss_vectorstore(chunks, embeddings)
else:
    pc = Pineconegrpc(api_key=PINECONE_API_KEY)
    index_name = "medassistpro"
    pc.create_index(
        name=index_name,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )
    vectorstore = Pinecone.from_documents(
        documents=chunks,
        index_name=index_name,
        embedding=embeddings,
    )
