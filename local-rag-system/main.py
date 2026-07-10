from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from app.api.endpoints import router
from app.rag.chain import RagService
from app.rag.vector_store import VectorStoreManager

app = FastAPI(title="Local RAG System", version="1.0.0")

# CORS 跨域配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """全局异常处理器，统一错误响应格式"""
    if isinstance(exc, HTTPException):
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "code": exc.status_code,
                "message": exc.detail,
                "detail": str(exc),
            },
        )
    return JSONResponse(
        status_code=500,
        content={
            "code": 500,
            "message": "服务器内部错误",
            "detail": str(exc),
        },
    )


@app.on_event("startup")
async def startup_event():
    """应用启动时初始化单例"""
    try:
        app.state.vector_store_manager = VectorStoreManager()
        print("VectorStoreManager initialized successfully")
        
        app.state.rag_service = RagService(vector_store_manager=app.state.vector_store_manager)
        print("RagService initialized successfully")
    except Exception as e:
        print(f"Failed to initialize services: {e}")
        raise


@app.get("/")
def root():
    return {"message": "Local RAG System is running"}


app.include_router(router, prefix="/api")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
