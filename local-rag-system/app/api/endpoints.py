from fastapi import APIRouter, UploadFile, File, HTTPException, Request
from pydantic import BaseModel
from app.rag.vector_store import VectorStoreManager
import os
from app.core.config import settings

router = APIRouter()


class ChatRequest(BaseModel):
    question: str


class SourceItem(BaseModel):
    source: str
    content: str


class ChatResponse(BaseModel):
    answer: str
    sources: list[SourceItem]


class UploadResponse(BaseModel):
    status: str
    filename: str
    chunks: int


class HealthResponse(BaseModel):
    status: str


class ErrorResponse(BaseModel):
    code: int
    message: str
    detail: str


@router.post("/chat", response_model=ChatResponse)
def chat(request: Request, body: ChatRequest):
    """用户提问，返回 RAG 回答"""
    try:
        rag_service = request.app.state.rag_service
        result = rag_service.chat(body.question)

        sources = [
            SourceItem(source=item["source"], content=item["content"])
            for item in result.get("sources", [])
        ]

        return ChatResponse(answer=result["answer"], sources=sources)

    except AttributeError:
        raise HTTPException(status_code=500, detail="RagService 未初始化")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/upload", response_model=UploadResponse)
def upload_document(request: Request, file: UploadFile = File(...)):
    """上传单个文档文件存入向量库"""
    try:
        file_path = os.path.join(settings.DATA_DIR, file.filename)

        with open(file_path, "wb") as buffer:
            buffer.write(file.file.read())

        manager = request.app.state.vector_store_manager
        result = manager.add_single_file(file_path)

        if result["success"]:
            return UploadResponse(
                status="success",
                filename=file.filename,
                chunks=result["chunks_added"],
            )
        else:
            raise HTTPException(status_code=400, detail=result["error"])

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health", response_model=HealthResponse)
def health_check():
    """健康检查，返回服务状态"""
    return HealthResponse(status="healthy")
