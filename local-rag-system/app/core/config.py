from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env")

    # Qdrant 向量数据库配置
    QDRANT_PATH: str = "./storage/qdrant_local"
    COLLECTION_NAME: str = "rag_knowledge_base"

    # Ollama 嵌入模型配置
    EMBEDDING_MODEL: str = "quentinz/bge-small-zh-v1.5"
    OLLAMA_BASE_URL: str = "http://localhost:11434"

    # Ollama 大语言模型配置
    OLLAMA_LLM_MODEL: str = "qwen2:7b"
    LLM_TEMPERATURE: float = 0.1

    # 文本切分配置
    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 50

    # 中文分隔符列表（用于文本切分）
    CHINESE_SEPARATORS: list[str] = [
        "\n\n",
        "\n",
        "。",
        "！",
        "？",
        "；",
        "，",
        "、",
        " ",
        "",
    ]

    # 数据目录配置
    DATA_DIR: str = "./data"

    # 检索配置
    RETRIEVER_TOP_K: int = 3


settings = Settings()
