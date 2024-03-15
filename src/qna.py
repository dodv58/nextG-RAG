from langchain_community.llms import LlamaCpp
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnablePick
from langchain_community.vectorstores import Chroma
from langchain.prompts import PromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate
from .common import format_docs

class LLM(object):
    def init_llm(self, app, embedding):
        self.embedding = embedding
        if app.config['DEVICE'] == 'cuda':
            self.llm = LlamaCpp(
                model_path=app.config['LLM_MODEL_PATH'],
                temperature=0.1,
                max_tokens=4096,
                top_p=1,
                n_gpu_layers=-1,
                n_batch=512,
                # callback_manager=callback_manager,
                n_ctx=2048,
                verbose=False,  # Verbose is required to pass to the callback manager
            )
        else:
            self.llm = LlamaCpp(
                model_path=app.config['LLM_MODEL'],
                temperature=0,
                max_tokens=4096,
                top_p=1,
                # callback_manager=callback_manager,
                n_ctx=2048,
                verbose=False,  # Verbose is required to pass to the callback manager
            )


    def qna(self, file, question):
        if file:
            vectorstore = Chroma(persist_directory=file[2], embedding_function=self.embedding.embedding,)

            template = """[INST]<<SYS>> Use the following pieces of context to answer the question at the end. If the provided context does not contain the answer, just say that you don't know, don't try to make up an answer. Use three sentences maximum and keep the answer as concise as possible.<</SYS>>
        
    Question: {question}
        
    Context: {context}
        
    Answer: [/INST]
    """
            rag_prompt_llama = PromptTemplate.from_template(template)
            # rag_prompt_llama = hub.pull("rlm/rag-prompt-llama")

            # Chain
            retriever = vectorstore.as_retriever()
            qa_chain = (
                    {"context": retriever | format_docs, "question": RunnablePassthrough()}
                    | rag_prompt_llama
                    | self.llm
                    | StrOutputParser()
            )
        else:
            template = """[INST]<<SYS>> Try to answer this question. If you don't know the answer, just say that you don't know, don't try to make up an answer. Use three sentences maximum and keep the answer as concise as possible.<</SYS>>
    
    Question: {question}
    
    Answer: [/INST]
    """
            rag_prompt_llama = PromptTemplate.from_template(template)
            # rag_prompt_llama = hub.pull("rlm/rag-prompt-llama")

            # Chain
            qa_chain = (
                    {"question": RunnablePassthrough()}
                    | rag_prompt_llama
                    | self.llm
                    | StrOutputParser()
            )

        return qa_chain.invoke(question)


