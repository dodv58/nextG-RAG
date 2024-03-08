from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import LlamaCpp
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_community.embeddings import LlamaCppEmbeddings

import argparse

def document_embedding(file, model=None):
    loader = PyPDFLoader(file)
    data = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    all_splits = text_splitter.split_documents(data)

    if not model:
        vectorstore = Chroma.from_documents(documents=all_splits, embedding=GPT4AllEmbeddings())
    else:
        vectorstore = Chroma.from_documents(documents=all_splits, embedding=LlamaCppEmbeddings(model_path=model))
    return vectorstore

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def summarize(llm, vectorstore, question):
    prompt = PromptTemplate.from_template(
        "Summarize the main themes in these retrieved docs: {docs}"
    )


    # Chain
    chain = {"docs": format_docs} | prompt | llm | StrOutputParser()

    # Run
    docs = vectorstore.similarity_search(question)
    answer = chain.invoke(docs)
    return answer



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", required=True, help="PDF file path")
    parser.add_argument("-m", help="model path", default="llama-2-7b-chat.Q4_K_M.gguf")
    args = parser.parse_args()


    vectorstore = document_embedding(args.f, model=args.m)

    #question = "What is multicast routing?"
    #docs = vectorstore.similarity_search(question)

    llm = LlamaCpp(
        model_path=args.m,
        temperature=0.1,
        max_tokens=4096,
        top_p=1,
        # callback_manager=callback_manager,
        n_ctx=2048,
        verbose=True,  # Verbose is required to pass to the callback manager
    )

    answer = summarize(llm, vectorstore, "what is weight optimization for multicast routing?")
    print(answer)



    rag_prompt = hub.pull("rlm/rag-prompt")


