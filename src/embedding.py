from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import LlamaCppEmbeddings, HuggingFaceEmbeddings
import chromadb
import uuid


def document_embedding(filepath, embedding=None):
    if filepath.rsplit('.', 1)[1].lower() == 'pdf':
        loader = PyPDFLoader(filepath)
    else:
        loader = TextLoader(filepath)
    data = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    all_splits = text_splitter.split_documents(data)

    chroma_dir = str(uuid.uuid4())
    vectorstore = Chroma.from_documents(documents=all_splits,
                                        embedding=embedding,
                                        persist_directory=f"./chromadb/{chroma_dir}")
    return chroma_dir

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def init_embedding(model=None):
    # return LlamaCppEmbeddings(model_path=f"./models/{model}")
    return HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

