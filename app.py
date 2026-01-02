import os
from typing import TypedDict

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain.prompts import PromptTemplate
from langgraph.graph import StateGraph

# -----------------------------
# Load PDFs
# -----------------------------
def load_documents():
    docs = []
    for file in os.listdir("data"):
        if file.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join("data", file))
            docs.extend(loader.load())
    return docs


# -----------------------------
# Build RAG Pipeline ONCE
# -----------------------------
docs = load_documents()

embeddings = OllamaEmbeddings(model="nomic-embed-text")
vectorstore = FAISS.from_documents(docs, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

llm = OllamaLLM(model="llama3.1:8b", temperature=0.2)

prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are a helpful assistant.
Answer ONLY using the context below.

Context:
{context}

Question:
{question}

Answer:
"""
)

class ChatState(TypedDict):
    question: str
    context: str
    answer: str


def retrieve(state: ChatState):
    docs = retriever.get_relevant_documents(state["question"])
    context = "\n".join(d.page_content for d in docs)
    return {"context": context}


def generate(state: ChatState):
    answer = llm.invoke(
        prompt.format(
            context=state["context"],
            question=state["question"]
        )
    )
    return {"answer": answer}


graph = StateGraph(ChatState)
graph.add_node("retrieve", retrieve)
graph.add_node("generate", generate)
graph.set_entry_point("retrieve")
graph.add_edge("retrieve", "generate")
graph.set_finish_point("generate")

rag_app = graph.compile()


# -----------------------------
# FUNCTION STREAMLIT WILL CALL
# -----------------------------
def ask_question(question: str, chat_history: str) -> str:
    combined_question = f"""
Conversation so far:
{chat_history}

Current question:
{question}
"""
    result = rag_app.invoke({"question": combined_question})
    return result["answer"]

