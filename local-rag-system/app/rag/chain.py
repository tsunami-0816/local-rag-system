from typing import List, Dict

from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_ollama import ChatOllama
from app.rag.vector_store import VectorStoreManager
from app.core.config import settings


class RagService:
    """RAG 问答服务，封装检索增强生成的完整链路"""

    def __init__(self, vector_store_manager: VectorStoreManager = None):
        """初始化大模型、检索器和问答链

        Args:
            vector_store_manager: 已初始化的向量存储管理器，若为 None 则新建
        """
        try:
            # 初始化 Ollama 大语言模型
            self._llm = ChatOllama(
                model=settings.OLLAMA_LLM_MODEL,
                temperature=settings.LLM_TEMPERATURE,
                base_url=settings.OLLAMA_BASE_URL,
            )
        except Exception as e:
            raise ConnectionError(f"无法连接到 Ollama 服务: {str(e)}")

        # 使用传入的或新建向量存储管理器
        self._vector_store_manager = vector_store_manager or VectorStoreManager()
        self._retriever = self._vector_store_manager.get_retriever()

        # 构建 LCEL 问答链
        self._chain = self._build_chain()

    def _format_docs(self, docs: List[Document]) -> str:
        """将检索到的文档列表格式化为上下文字符串

        Args:
            docs: 文档对象列表

        Returns:
            格式化后的上下文文本
        """
        if not docs:
            return ""
        return "\n\n".join(doc.page_content for doc in docs)

    def _build_chain(self):
        """构建 LCEL 问答链

        LCEL 链路：检索器 → 格式化文档 → 提示词 → 大模型 → 字符串输出

        Returns:
            LCEL Runnable 链
        """
        # 提示词模板：明确要求基于上下文回答，不知道就返回固定短语
        template = """你是一个专业的问答助手。请根据以下提供的上下文信息回答用户的问题。

重要规则：
1. 你的回答必须完全基于提供的上下文信息，不得使用任何外部知识
2. 如果上下文信息不足以回答问题，或者问题与上下文无关，请直接回答「根据现有资料无法回答该问题」
3. 回答要简洁准确，使用中文输出
4. 不要在回答中提及「根据上下文」、「资料中」等字样，直接给出答案

上下文：
{context}

问题：{question}"""

        prompt = ChatPromptTemplate.from_template(template)

        # LCEL 链式调用
        chain = (
            {
                "context": self._retriever | self._format_docs,
                "question": RunnablePassthrough(),
            }
            | prompt
            | self._llm
            | StrOutputParser()
        )

        return chain

    def _deduplicate_sources(self, docs: List[Document]) -> List[Dict]:
        """对检索到的文档来源进行去重（按文件级别）

        Args:
            docs: 文档对象列表

        Returns:
            去重后的来源列表，包含文件名和内容片段
        """
        seen_sources = {}
        for doc in docs:
            source = doc.metadata.get("source", "未知来源")
            if source not in seen_sources:
                seen_sources[source] = {
                    "source": source,
                    "content": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                }
        return list(seen_sources.values())

    def chat(self, question: str) -> Dict:
        """接收用户问题，返回回答内容和检索到的来源片段

        Args:
            question: 用户问题

        Returns:
            包含回答和来源信息的字典
        """
        try:
            # 获取检索到的文档（用于返回来源）
            retrieved_docs = self._retriever.invoke(question)

            # 调用问答链获取回答
            answer = self._chain.invoke(question)

            # 对来源进行去重
            sources = self._deduplicate_sources(retrieved_docs)

            return {
                "answer": answer,
                "sources": sources,
                "retrieved_count": len(retrieved_docs),
            }

        except ConnectionError as e:
            return {
                "answer": "根据现有资料无法回答该问题",
                "sources": [],
                "retrieved_count": 0,
                "error": str(e),
            }
        except Exception as e:
            return {
                "answer": "根据现有资料无法回答该问题",
                "sources": [],
                "retrieved_count": 0,
                "error": f"问答过程中发生错误: {str(e)}",
            }
