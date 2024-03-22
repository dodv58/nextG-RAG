from langchain_community.llms import LlamaCpp
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnablePick
from langchain_community.vectorstores import Chroma
from langchain.prompts import PromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate
from .common import format_docs

template = {
    "rag": {
        "en": """[INST]<<SYS>> Use the following pieces of context to answer the question at the end. If the provided context does not contain the answer, just say that you don't know, don't try to make up an answer. Use three sentences maximum and keep the answer as concise as possible.<</SYS>>
        
    Question: {question}
        
    Context: {context}
        
    Answer: [/INST]
    """,
        "vi": """[INST]<<SYS>> Sử dụng các phần ngữ cảnh sau đây để trả lời câu hỏi ở cuối. Nếu ngữ cảnh được cung cấp không chứa câu trả lời, bạn chỉ cần nói rằng bạn không biết, đừng cố bịa ra câu trả lời. Sử dụng tối đa ba câu và giữ câu trả lời ngắn gọn nhất có thể.<</SYS>>

                Câu hỏi: {question}

                Ngữ cảnh: {context}

                Trả lời: [/INST]
                """
    },
    "general": {
        "en": """[INST]<<SYS>> Try to answer this question. If you don't know the answer, just say that you don't know, don't try to make up an answer. Use three sentences maximum and keep the answer as concise as possible.<</SYS>>
    
    Question: {question}
    
    Answer: [/INST]
    """,
        "vi": """[INST]<<SYS>> Hãy cố gắng trả lời câu hỏi này. Nếu không biết câu trả lời, bạn chỉ cần nói rằng bạn không biết, đừng cố bịa ra câu trả lời. Sử dụng tối đa ba câu và giữ câu trả lời ngắn gọn nhất có thể.<</SYS>>

                Câu hỏi: {question}

                Trả lời: [/INST]
                """
    }
}

class LLM(object):
    def init_llm(self, app, embedding):
        print("Initializing LLM")
        self.embedding = embedding
        self.lang = "vi" if "vietnamese" in app.config["LLM_MODEL"] else "en"
        if app.config['DEVICE'].type == 'cuda':
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
        elif app.config['DEVICE'].type == 'mps':
            self.llm = LlamaCpp(
                model_path=app.config['LLM_MODEL_PATH'],
                n_gpu_layers=-1,
                n_batch=64,
                f16_kv=True,  # MUST set to True, otherwise you will run into problem after a couple of calls
                n_ctx=2048,
                temperature=0,
                max_tokens=4096,
                verbose=False,  # Verbose is required to pass to the callback manager
            )
        else:
            self.llm = LlamaCpp(
                model_path=app.config['LLM_MODEL_PATH'],
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

            rag_prompt_llama = PromptTemplate.from_template(template["rag"][self.lang])
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
            rag_prompt_llama = PromptTemplate.from_template(template['general'][self.lang])
            # rag_prompt_llama = hub.pull("rlm/rag-prompt-llama")

            # Chain
            qa_chain = (
                    {"question": RunnablePassthrough()}
                    | rag_prompt_llama
                    | self.llm
                    | StrOutputParser()
            )

        return qa_chain.invoke(question)


