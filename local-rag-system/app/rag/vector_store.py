import os
from typing import List, Optional

from qdrant_client import QdrantClient
from langchain_qdrant import QdrantVectorStore
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    Docx2txtLoader,
)
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever

from app.core.config import settings


class VectorStoreManager:
    """向量存储管理器，封装 Qdrant 向量库的初始化、文档添加和检索功能"""

    def __init__(self):
        """初始化向量库和嵌入模型"""
        # 初始化 Qdrant 客户端（本地磁盘持久化模式）
        self._client = QdrantClient(path=settings.QDRANT_PATH)

        # 初始化 Ollama 嵌入模型
        self._embeddings = OllamaEmbeddings(
            model=settings.EMBEDDING_MODEL,
            base_url=settings.OLLAMA_BASE_URL,
        )

        # 检查集合是否存在，不存在则先创建
        if not self._client.collection_exists(settings.COLLECTION_NAME):
            # 获取嵌入维度（通过调用一次嵌入模型获取）
            try:
                sample_embedding = self._embeddings.embed_query("测试")
                embedding_dim = len(sample_embedding)
            except Exception:
                embedding_dim = 384

            # 创建集合
            self._client.create_collection(
                collection_name=settings.COLLECTION_NAME,
                vectors_config={
                    "default": {
                        "size": embedding_dim,
                        "distance": "Cosine",
                    }
                },
            )

        # 初始化 Qdrant 向量存储
        self._vector_store = QdrantVectorStore(
            client=self._client,
            collection_name=settings.COLLECTION_NAME,
            embedding=self._embeddings,
            vector_name="default",
        )

        # 初始化中文文本切分器
        self._text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
            separators=settings.CHINESE_SEPARATORS,
            length_function=len,
        )

    def _ensure_collection_exists(self):
        """确保集合存在，不存在则创建"""
        if not self._client.collection_exists(settings.COLLECTION_NAME):
            try:
                sample_embedding = self._embeddings.embed_query("测试")
                embedding_dim = len(sample_embedding)
            except Exception:
                embedding_dim = 384

            self._client.create_collection(
                collection_name=settings.COLLECTION_NAME,
                vectors_config={
                    "default": {
                        "size": embedding_dim,
                        "distance": "Cosine",
                    }
                },
            )

    def _load_document(self, file_path: str) -> List[Document]:
        """根据文件类型加载文档内容

        Args:
            file_path: 文件路径

        Returns:
            文档对象列表

        Raises:
            FileNotFoundError: 文件不存在
            ValueError: 不支持的文件格式
            Exception: 其他加载错误
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"文件不存在: {file_path}")

        file_ext = os.path.splitext(file_path)[1].lower()

        if file_ext == ".txt":
            loader = TextLoader(file_path, encoding="utf-8")
        elif file_ext == ".pdf":
            loader = PyPDFLoader(file_path)
        elif file_ext == ".docx":
            loader = Docx2txtLoader(file_path)
        else:
            raise ValueError(f"不支持的文件格式: {file_ext}")

        try:
            return loader.load()
        except Exception as e:
            raise Exception(f"加载文件失败 {file_path}: {str(e)}")

    def _split_documents(self, documents: List[Document]) -> List[Document]:
        """切分文档为小片段

        Args:
            documents: 原始文档对象列表

        Returns:
            切分后的文档片段列表
        """
        if not documents:
            return []
        return self._text_splitter.split_documents(documents)

    def add_documents_from_dir(self, dir_path: str) -> dict:
        """加载指定目录下所有支持的文档，切分后入库

        Args:
            dir_path: 目录路径

        Returns:
            统计信息字典，包含成功/失败数量和详细信息
        """
        if not os.path.isdir(dir_path):
            raise FileNotFoundError(f"目录不存在: {dir_path}")

        results = {
            "total_files": 0,
            "success_count": 0,
            "failed_count": 0,
            "total_chunks": 0,
            "success_files": [],
            "failed_files": [],
        }

        supported_extensions = {".txt", ".pdf", ".docx"}

        for filename in os.listdir(dir_path):
            file_path = os.path.join(dir_path, filename)

            if not os.path.isfile(file_path):
                continue

            file_ext = os.path.splitext(filename)[1].lower()
            if file_ext not in supported_extensions:
                continue

            results["total_files"] += 1

            try:
                # 加载文档
                docs = self._load_document(file_path)
                # 切分文档
                split_docs = self._split_documents(docs)

                if split_docs:
                    # 确保集合存在
                    self._ensure_collection_exists()
                    # 入库
                    self._vector_store.add_documents(split_docs)
                    results["success_count"] += 1
                    results["total_chunks"] += len(split_docs)
                    results["success_files"].append(
                        {"filename": filename, "chunks": len(split_docs)}
                    )
                else:
                    results["failed_count"] += 1
                    results["failed_files"].append(
                        {"filename": filename, "error": "文档内容为空或无法切分"}
                    )

            except Exception as e:
                results["failed_count"] += 1
                results["failed_files"].append(
                    {"filename": filename, "error": str(e)}
                )

        return results

    def add_single_file(self, file_path: str) -> dict:
        """接收单个文件，解析后入库

        Args:
            file_path: 文件路径

        Returns:
            处理结果字典，包含文件名、切分数量等信息
        """
        try:
            # 加载文档
            docs = self._load_document(file_path)

            if not docs:
                return {
                    "success": False,
                    "filename": os.path.basename(file_path),
                    "error": "文档内容为空",
                }

            # 切分文档
            split_docs = self._split_documents(docs)

            if not split_docs:
                return {
                    "success": False,
                    "filename": os.path.basename(file_path),
                    "error": "文档无法切分为有效片段",
                }

            # 确保集合存在
            self._ensure_collection_exists()
            # 入库
            self._vector_store.add_documents(split_docs)

            return {
                "success": True,
                "filename": os.path.basename(file_path),
                "original_docs": len(docs),
                "chunks_added": len(split_docs),
            }

        except FileNotFoundError as e:
            return {
                "success": False,
                "filename": os.path.basename(file_path),
                "error": str(e),
            }
        except ValueError as e:
            return {
                "success": False,
                "filename": os.path.basename(file_path),
                "error": str(e),
            }
        except Exception as e:
            return {
                "success": False,
                "filename": os.path.basename(file_path),
                "error": f"处理文件失败: {str(e)}",
            }

    def get_retriever(self) -> VectorStoreRetriever:
        """返回 LangChain 检索器，返回 top 3 最相似片段

        Returns:
            VectorStoreRetriever 对象
        """
        return self._vector_store.as_retriever(
            search_kwargs={"k": settings.RETRIEVER_TOP_K}
        )

    def clear_collection(self):
        """清空向量库集合"""
        if self._client.collection_exists(settings.COLLECTION_NAME):
            self._client.delete_collection(settings.COLLECTION_NAME)
