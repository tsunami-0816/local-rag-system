Local RAG System
一个基于 LangChain 和 Ollama 的本地 RAG（检索增强生成）知识库问答系统，支持文档上传、向量化存储和智能问答。

✨ 功能特性
本地部署：所有数据存储在本地，无需联网即可使用
多格式支持：支持 PDF、DOCX、TXT 文档上传
中文优化：使用中文嵌入模型和中文提示词
来源追溯：回答附带检索到的来源片段，提高可信度
API 接口：提供 RESTful API，方便集成到其他系统
Swagger UI：自动生成接口文档，便于调试

🛠️ 技术栈
组件	技术
语言	Python 3.10+
框架	FastAPI
向量数据库	Qdrant（本地持久化）
嵌入模型	quentinz/bge-small-zh-v1.5
大语言模型	qwen2:7b
向量化工具	LangChain

🚀 快速开始
环境要求
Python 3.10+
Ollama（用于运行本地大模型）

安装步骤
# 1. 克隆仓库
git clone https://github.com/你的用户名/local-rag-system.git
cd local-rag-system

# 2. 创建虚拟环境
python -m venv .venv

# 3. 激活虚拟环境
# Windows
.venv\Scripts\Activate.ps1
# macOS/Linux
source .venv/bin/activate

# 4. 安装依赖
pip install -r requirements.txt

启动 Ollama
# 下载所需模型
ollama pull quentinz/bge-small-zh-v1.5
ollama pull qwen2:7b

# 启动 Ollama 服务
ollama serve

启动服务
python main.py

服务启动后访问：

API 地址：http://localhost:8000
Swagger 文档：http://localhost:8000/docs

📡 API 接口

健康检查
GET /api/health

上传文档
POST /api/upload
Content-Type: multipart/form-data

# 参数
file: 文档文件（支持 .pdf, .docx, .txt）

响应示例：
{
  "status": "success",
  "filename": "document.pdf",
  "chunks": 15
}

智能问答
POST /api/chat
Content-Type: application/json

# 请求体
{
  "question": "你的问题"
}

响应示例：
{
  "answer": "回答内容...",
  "sources": [
    {
      "source": "./data/document.pdf",
      "content": "相关片段内容..."
    }
  ]
}


📁 项目结构
local-rag-system/
├── app/                    # 应用代码
│   ├── api/               # API 接口层
│   │   └── endpoints.py   # 路由定义
│   ├── core/              # 配置层
│   │   └── config.py      # 配置参数
│   ├── rag/               # RAG 核心
│   │   ├── vector_store.py # 向量存储管理
│   │   └── chain.py       # 问答链
│   └── utils/             # 工具函数
│       └── document_loader.py # 文档加载
├── data/                  # 测试文档目录
├── storage/               # Qdrant 数据持久化
├── main.py                # 启动入口
├── test_demo.py           # 测试脚本
├── requirements.txt       # 依赖列表
└── .gitignore             # Git 忽略文件

⚙️ 配置说明
配置文件位于 app/core/config.py：

配置项	默认值	说明
QDRANT_PATH	./storage/qdrant_local	向量库存储路径
COLLECTION_NAME	rag_knowledge_base	集合名称
EMBEDDING_MODEL	quentinz/bge-small-zh-v1.5	嵌入模型
OLLAMA_LLM_MODEL	qwen2:7b	大语言模型
LLM_TEMPERATURE	0.1	模型温度
CHUNK_SIZE	500	文本切分大小
CHUNK_OVERLAP	50	切分重叠大小

🧪 测试
运行测试脚本：
python test_demo.py

测试脚本会自动：
上传 data 目录下的测试文档
测试问答功能
打印回答和来源

📝 使用示例
上传文档
curl.exe -X POST "http://localhost:8000/api/upload" -F "file=@your_document.pdf"
提问
curl.exe -X POST "http://localhost:8000/api/chat" ^
  -H "Content-Type: application/json" ^
  -d "{\"question\":\"什么是 RAG？\"}"

