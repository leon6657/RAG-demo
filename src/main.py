"""CLI 入口：ingest / ask / chat 命令"""

import logging
from pathlib import Path

import click
from dotenv import load_dotenv

from src.config import Settings
from src.ingest import IngestionPipeline
from src.retrieve import Retriever
from src.generate import Generator

# 加载 .env 文件中的环境变量
env_path = Path(__file__).resolve().parent.parent / ".env"
if env_path.exists():
    load_dotenv(env_path)

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


@click.group()
def cli():
    """AI 知识库 —— 基于 RAG 的技术问答系统"""


@cli.command()
def ingest():
    """读取 Markdown 笔记并重建向量知识库"""
    settings = Settings()
    pipeline = IngestionPipeline(settings)
    pipeline.run()
    click.echo("知识库构建完成！")


@cli.command()
@click.argument("query")
def ask(query: str):
    """单次问答：检索并生成回答"""
    settings = Settings()
    retriever = Retriever(settings)
    generator = Generator(settings)

    with_spans = retriever.retrieve(query)
    click.echo(f"\n{'='*60}")
    click.echo(f"问题：{query}")
    click.echo(f"{'='*60}\n")
    answer = generator.answer(query, with_spans)
    click.echo(answer)
    click.echo(f"\n{'='*60}")
    click.echo("来源文档：")
    for i, (doc, score) in enumerate(with_spans, 1):
        source = doc.metadata.get("source", "unknown")
        click.echo(f"  {i}. {Path(source).name} (相似度: {score:.4f})")


@cli.command()
def chat():
    """交互式多轮问答"""
    settings = Settings()
    retriever = Retriever(settings)
    generator = Generator(settings)

    click.echo("进入交互模式（输入 exit 退出）\n")
    while True:
        query = click.prompt("问题", prompt_suffix="> ")
        if query.lower() in ("exit", "quit", "q"):
            break
        if not query.strip():
            continue

        with_spans = retriever.retrieve(query)
        click.echo()
        answer = generator.answer(query, with_spans)
        click.echo(answer)
        click.echo()


if __name__ == "__main__":
    cli()
