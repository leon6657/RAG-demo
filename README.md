# AI 知识库 - 基于 RAG 的技术问答系统

从零搭建的最小 RAG 知识库项目，面向 AI 实习岗位（如 ByteDance 飞书项目）的简历项目。

路线：最小 RAG -> 优化 RAG -> Agent -> LangGraph

设计文档：docs/rag-knowledge-base-design.md

## 技术栈

- LLM：DeepSeek Chat API（OpenAI 兼容接口）
- Embedding：BAAI/bge-small-zh-v1.5（本地）或 DeepSeek Embedding API
- 向量库：ChromaDB（本地持久化）
- 框架：LangChain + LangChain-Chroma
- CLI：Click

## 快速开始

1. 设置环境变量

复制 .env.template 为 .env，填入 DeepSeek API Key，或设置环境变量：

$env:DEEPSEEK_API_KEY="sk-your-key"

2. 安装依赖

python -m venv .venv
.venv\Scripts\pip install -r requirements.txt

3. 构建知识库

.venv\Scripts\python src/main.py ingest

4. 开始问答

.venv\Scripts\python src/main.py ask "什么是RAG"

或进入交互模式：

.venv\Scripts\python src/main.py chat

## 命令参考

- python src/main.py ingest - 读取 data/notes/ 下的笔记，分块后存入向量库
- python src/main.py ask "问题" - 单次问答，显示来源与相似度
- python src/main.py chat - 交互式多轮问答

## 项目结构

ai-knowledge-base/
- data/notes/ - Markdown 知识笔记
- data/chroma_db/ - ChromaDB 持久化目录（自动生成）
- src/config.py - 全局配置
- src/_embeddings.py - Embedding 工厂
- src/ingest.py - 文档加载、分块、Embedding、入库
- src/retrieve.py - 查询向量化与 ChromaDB 检索
- src/generate.py - 检索结果 + Prompt -> LLM 回答
- src/main.py - CLI 入口
- tests/ - 集成测试
- docs/ - 设计文档和实现计划

## Embedding 方式切换

编辑 src/config.py 中的 use_api_embedding：
- True（默认）：使用 DeepSeek Embedding API（需网络和 API Key）
- False：使用本地 BGE 模型（首次需下载模型到 ~/.cache/huggingface/）

## 路线图

Phase 1 - 最小 RAG (已完成)：文档加载、分块、Embedding、检索、生成
Phase 2 - 优化 RAG (待开始)：重排序、查询改写、混合检索、评测
Phase 3 - Agent (计划中)：知识库工具化、对话记忆、ReAct 循环
Phase 4 - LangGraph (计划中)：有向状态图、条件路由、会话管理

## 运行测试

$env:DEEPSEEK_API_KEY="sk-your-key"
.venv\Scripts\pytest tests/ -v

测试覆盖：
- test_config_defaults - 验证配置默认值
- test_config_paths - 验证路径解析
- test_ingestion_creates_chroma_db - 需要 API Key
- test_retrieve_returns_docs - 需要 API Key
