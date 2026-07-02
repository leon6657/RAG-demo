"""LLM 生成层：基于检索结果构造 Prompt 并调用 LLM"""

import logging

from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage

from src.config import Settings

logger = logging.getLogger(__name__)

RAG_SYSTEM_PROMPT = """你是一个基于知识库的技术问答助手。

请根据以下上下文信息回答用户的问题。
上下文信息中包含了相关的技术笔记内容。
如果上下文中没有足够的信息来回答问题，请明确告诉用户你不知道，不要编造答案。
用中文回答，保持回答简洁、准确、有技术深度。"""


class Generator:
    """基于检索上下文 + 用户问题调用 LLM 生成回答"""

    def __init__(self, settings: Settings):
        self.settings = settings
        self._llm = ChatOpenAI(
            model=settings.llm_model,
            openai_api_key=settings.deepseek_api_key,
            openai_api_base=settings.llm_base_url,
            temperature=settings.llm_temperature,
        )

    def answer(self, query: str, context_docs: list) -> str:
        """生成回答"""
        context_parts = []
        for i, (doc, score) in enumerate(context_docs, 1):
            source = doc.metadata.get("source", "unknown")
            context_parts.append(
                f"[文档 {i}]（来源：{source}，相似度：{score:.4f}）\n{doc.page_content}"
            )
        context_str = "\n\n".join(context_parts)
        user_message = f"""上下文信息：
{context_str}

用户问题：{query}"""

        messages = [
            SystemMessage(content=RAG_SYSTEM_PROMPT),
            HumanMessage(content=user_message),
        ]
        logger.info(
            "Sending prompt to %s (context: %d docs)",
            self.settings.llm_model,
            len(context_docs),
        )
        response = self._llm.invoke(messages)
        return response.content
