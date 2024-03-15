from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import LlamaCppEmbeddings, HuggingFaceEmbeddings
import chromadb
import uuid
from .common import format_docs

class Embedding(object):
    def init_embedding(self, app):
        # return LlamaCppEmbeddings(model_path=f"./models/{model}")
        self.chromadb = app.config['VECTOR_DB_PATH']
        self.embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

    def document_embedding(self, filepath):
        if filepath.rsplit('.', 1)[1].lower() == 'pdf':
            loader = PyPDFLoader(filepath)
        else:
            loader = TextLoader(filepath)
        data = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        all_splits = text_splitter.split_documents(data)

        chroma_dir = f"{self.chromadb}/{uuid.uuid4()}"
        vectorstore = Chroma.from_documents(documents=all_splits,
                                            embedding=self.embedding,
                                            persist_directory=chroma_dir)
        return chroma_dir




