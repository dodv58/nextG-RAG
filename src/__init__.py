from .embedding import Embedding
from .qna import LLM

llm = LLM()
embedding = Embedding()

def init_app(app):
    embedding.init_embedding(app)
    llm.init_llm(app, embedding)