from langchain_community.llms import LlamaCpp
from langchain_core.output_parsers import StrOutputParser
from langchain import hub
from langchain_core.runnables import RunnablePassthrough, RunnablePick
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import LlamaCppEmbeddings
from langchain.prompts import PromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def qna(model, file, question):
    vectorstore = Chroma(persist_directory=file[2], embedding_function=LlamaCppEmbeddings(model_path=model),)
    # rag_prompt_llama = hub.pull("rlm/rag-prompt-llama")

    # rag_prompt_llama = PromptTemplate.from_template('[INST]<<SYS>> You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If the context does not provide the answer to the question, the assistant will say, "I am sorry, but I do not know the answer to that question". The assistant will not invent anything that is not drawn directly from the context and keep the answer concise.<</SYS>> \nQuestion: {question} \nContext: {context} \nAnswer: [/INST]')
    # rag_prompt_llama = HumanMessagePromptTemplate.from_template(rag_prompt_llama.template)
    # rag_prompt_llama = ChatPromptTemplate.from_messages([rag_prompt_llama])

    template = """[INST]<<SYS>> Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer. Use three sentences maximum and keep the answer as concise as possible.<</SYS>> 
    
    Question: {question} 
    
    Context: {context} 
    
    Answer: [/INST]
    """
    rag_prompt_llama = PromptTemplate.from_template(template)

    llm = LlamaCpp(
        model_path=model,
        temperature=0,
        max_tokens=4096,
        top_p=1,
        # callback_manager=callback_manager,
        n_ctx=2048,
        verbose=False,  # Verbose is required to pass to the callback manager
    )
    # Chain
    retriever = vectorstore.as_retriever()
    qa_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | rag_prompt_llama
            | llm
            | StrOutputParser()
    )

    # Run
    return qa_chain.invoke(question)


