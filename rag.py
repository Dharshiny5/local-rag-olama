from langchain_community.document_loaders import PyPDFLoader
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os 

###Loading Documents###
documents = []

for file in os.listdir("data"):
    if file.endswith("pdf"):
        loader = PyPDFLoader(f"data/{file}")
        documents.extend(loader.load())

print(f"Loaded {len(documents)} pages")

###Splitting text into chunks###
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)

chunks = splitter.split_documents(documents)
print(f"Created {len(chunks)} chunks")

###Create embeddings###
embeddings = OllamaEmbeddings(model="nomic-embed-text")

###Store in Vector database###
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db"
)

###Load LLM###
llm = OllamaLLM(model="llama3.2")

###Create Retriever###
retriever =vectorstore.as_retriever(search_kwargs = {"k":3})

###Query Loop###
while True:
    query = input("\nAsk a question (type 'exit' to quit): ")
    if query.lower() == "exit":
        break

    retrieved_docs = retriever.invoke(query)
    context = "\n\n" .join(doc.page_content for doc in retrieved_docs)

    prompt = f"""
Answer ONLY from the context below.
If the answer is not found, say "I don't know".

Context:
{context}

Question:
{query}
"""

 
    answer = llm.invoke(prompt)
    print("\nAnswer:\n", answer)
    print(answer)

