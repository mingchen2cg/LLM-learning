"""
第1章：环境安装与配置 - 基础示例
FastAPI 最简单的应用程序
"""

from fastapi import FastAPI

# 创建 FastAPI 应用实例
app = FastAPI(
    title="FastAPI 学习项目",
    description="这是一个 FastAPI 学习教程项目",
    version="1.0.0",
)


@app.get("/")
async def read_root():
    """根路径，返回欢迎消息"""
    return {"message": "Hello FastAPI! 欢迎学习 FastAPI 框架"}


@app.get("/health")
async def health_check():
    """健康检查接口"""
    return {"status": "healthy", "service": "健康检查"}


if __name__ == "__main__":
    import uvicorn

    # 开发模式启动，支持热重载
    uvicorn.run("main:app", host="127.0.0.1", port=8001, reload=True)
