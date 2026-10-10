"""
Medical RAG系统配置文件
"""

from dataclasses import asdict, dataclass
from typing import Dict, Any


@dataclass
class MedicalRAGConfig:
    """医疗RAG系统配置类"""

    # 路径配置
    data_path: str = "../data" # 医学md路径
    index_save_path: str = "../vector_index" # 向量索引路径 

    # 模型配置
    embedding_model: str = "BAAI/bge-small-zh-v1.5" 
    llm_model: str = "deepseek-v4-flash"

    # 检索配置
    top_k: int = 3 # 给llm3段学习资料
    rerank_enabled: bool = True
    reranker_model: str = "Qwen/Qwen3-Reranker-0.6B"
    rerank_candidate_k: int = 20
    rerank_batch_size: int = 4
    rerank_max_chars: int = 1500
    reranker_device: str | None = None  # None 时由 sentence-transformers 自动选择设备

    # 生成配置
    temperature: float = 0.3 # 回答随机程度
    max_tokens: int = 2048 # 回答最大长度

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'MedicalRAGConfig':
        """从字典创建配置对象"""
        return cls(**config_dict)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return asdict(self)


# 默认配置实例
DEFAULT_CONFIG = MedicalRAGConfig()
