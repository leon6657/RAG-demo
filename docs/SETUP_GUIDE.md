# Phase 1 最小 RAG 知识库 -- 搭建说明文档

> 项目路径：ai-knowledge-base/ 

> 技术栈：Python 3.9+ / LangChain / FAISS / BGE / DeepSeek Chat



## 一、项目概况

一个从零搭建的最小 RAG 知识库系统，面向 AI 实习岗位简历项目。

路线：最小 RAG -> 优化 RAG -> Agent -> LangGraph



## 二、技术选型



| 功能 | 方案 |

|------|------|

| Embedding | BAAI/bge-small-zh-v1.5（本地 512 维）|

| 向量索引 | FAISS |

| LLM 生成 | DeepSeek Chat |

| 分块 | RecursiveCharacterTextSplitter (500/50) |

| CLI | Click (ingest/ask/chat) |



## 三、搭建遇到的问题



1. DeepSeek 无 Embedding API -> 改用本地 BGE

2. HuggingFace 国内访问不了 -> hf-mirror.com 镜像

3. ChromaDB protobuf 冲突崩溃 -> 替换为 FAISS



## 四、快速上手



1. 配置 API Key：复制 .env.template 为 .env，填入 DEEPSEEK_API_KEY

2. 构建索引：python -m src.main ingest

3. 问答：python -m src.main ask "什么是RAG"

4. 测试：pytest tests/ -v（预期 4 passed）



## 五、参考链接

- 设计文档：docs/2026-07-02-rag-knowledge-base-design.md

- 实现计划：docs/superpowers/plans/2026-07-02-rag-phase1.md

- LangChain: https://python.langchain.com

- DeepSeek API: https://api-docs.deepseek.com/

