#!/usr/bin/env python3
"""
RAG 系统测试脚本（通过 API 接口操作）
功能：通过 HTTP API 测试文档入库和问答功能
要求：FastAPI 服务必须已经在 http://localhost:8000 运行
"""

import json
import os
import requests

API_BASE_URL = "http://localhost:8000/api"


def check_service():
    """检查服务是否运行"""
    try:
        response = requests.get(f"{API_BASE_URL}/health")
        if response.status_code == 200:
            print("[INFO] FastAPI 服务运行正常")
            return True
    except requests.exceptions.ConnectionError:
        print("[ERROR] FastAPI 服务未启动！请先运行 python main.py")
    return False


def upload_documents():
    """通过 API 上传 data 目录下的所有文档"""
    print("\n" + "=" * 60)
    print("步骤 1: 上传测试文档")
    print("=" * 60)

    data_dir = "./data"
    files = [f for f in os.listdir(data_dir) if f.endswith(('.txt', '.pdf', '.docx'))]

    if not files:
        print("[WARN] data 目录下没有找到支持的文档")
        return

    for filename in files:
        file_path = os.path.join(data_dir, filename)
        print(f"\n[INFO] 上传文件: {filename}")

        try:
            with open(file_path, "rb") as f:
                response = requests.post(
                    f"{API_BASE_URL}/upload",
                    files={"file": (filename, f)}
                )

            if response.status_code == 200:
                result = response.json()
                print(f"  ✅ 成功 - 入库 {result['chunks']} 个片段")
            else:
                print(f"  ❌ 失败 - {response.json().get('message', '未知错误')}")

        except Exception as e:
            print(f"  ❌ 异常 - {str(e)}")


def test_qa():
    """通过 API 测试问答功能"""
    print("\n" + "=" * 60)
    print("步骤 2: 测试问答功能")
    print("=" * 60)

    test_questions = [
        "什么是 RAG？它有哪些优势？",
        "常用的大语言模型有哪些？",
        "Qdrant 是什么？它有哪些特性？",
    ]

    for idx, question in enumerate(test_questions, 1):
        print(f"\n--- 问题 {idx}: {question} ---")

        try:
            response = requests.post(
                f"{API_BASE_URL}/chat",
                json={"question": question}
            )

            if response.status_code == 200:
                result = response.json()
                answer = result["answer"]
                sources = result.get("sources", [])

                print(f"\n【回答】:\n{answer}")

                if sources:
                    print(f"\n【来源 ({len(sources)} 个)】:")
                    for i, source in enumerate(sources, 1):
                        print(f"\n  [{i}] 文件: {source['source']}")
                        print(f"     片段: {source['content'][:150]}...")
                else:
                    print("\n【来源】: 无")

            else:
                print(f"❌ 问答失败 - {response.json().get('message', '未知错误')}")

        except Exception as e:
            print(f"❌ 异常 - {str(e)}")


if __name__ == "__main__":
    print("本地 RAG 知识库问答系统 - 测试脚本（API 版）")
    print("=" * 60)
    print("要求：FastAPI 服务必须已在 http://localhost:8000 运行")

    if not check_service():
        exit(1)

    try:
        upload_documents()
        test_qa()

        print("\n" + "=" * 60)
        print("测试完成！")
        print("=" * 60)

    except KeyboardInterrupt:
        print("\n[INFO] 用户中断，测试结束")
    except Exception as e:
        print(f"\n[ERROR] 测试过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
