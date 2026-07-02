"""应用全局配置"""

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Settings:
    notes_dir: Path = Path("data/notes")
    chroma_dir: Path = Path("data/chroma_db")
    embedding_model: str = "BAAI/bge-small-zh-v1.5"
    chunk_size: int = 500
    chunk_overlap: int = 50
    top_k: int = 4
    llm_model: str = "deepseek-chat"
    llm_base_url: str = "https://api.deepseek.com/v1"
    llm_temperature: float = 0.3
    deepseek_api_key: str = ""

    def __post_init__(self):
        if not self.deepseek_api_key:
            self.deepseek_api_key = os.environ.get("DEEPSEEK_API_KEY", "")

    def project_root(self) -> Path:
        return Path(__file__).resolve().parent.parent

    def resolved_notes_dir(self) -> Path:
        return self.project_root() / self.notes_dir

    def resolved_chroma_dir(self) -> Path:
        return self.project_root() / self.chroma_dir
