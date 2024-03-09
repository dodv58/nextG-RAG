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

    template = """[INST]<<SYS>> Use the following pieces of context to answer the question at the end. If the provided context does not contain the answer, just say that you don't know, don't try to make up an answer. Use three sentences maximum and keep the answer as concise as possible.<</SYS>>

Question: {question}

Context: {context}

Answer: [/INST]
"""
    rag_prompt_llama = PromptTemplate.from_template(template)
    # rag_prompt_llama = hub.pull("rlm/rag-prompt-llama")

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


