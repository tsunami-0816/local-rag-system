from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from app.rag.vector_store import VectorStoreManager
from app.core.config import settings


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def get_qa_chain():
    llm = ChatOllama(model=settings.OLLAMA_MODEL)
    manager = VectorStoreManager()
    retriever = manager.get_retriever()

    template = """根据以下上下文回答问题：

{context}

问题：{question}
"""
    prompt = ChatPromptTemplate.from_template(template)

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain, retriever


def ask_question(question: str) -> dict:
    chain, retriever = get_qa_chain()
    answer = chain.invoke(question)
    docs = retriever.get_relevant_documents(question)

    return {
        "answer": answer,
        "sources": [
            doc.metadata.get("source", "") for doc in docs
        ],
    }
