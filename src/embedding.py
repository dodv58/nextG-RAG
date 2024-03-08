from langchain_community.embeddings import LlamaCppEmbeddings
llama = LlamaCppEmbeddings(model_path="llama-2-13b-chat.Q6_K.gguf")
text = "This is a test document."
query_result = llama.embed_query(text)
doc_result = llama.embed_documents([text])
