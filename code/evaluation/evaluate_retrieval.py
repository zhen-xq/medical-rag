"""
检索评测脚本 - 只评估“问题是否召回正确资料”，不调用生成模型。

运行方式（在 code/ 目录下）：
    python evaluation/evaluate_retrieval.py
    python evaluation/evaluate_retrieval.py --compare  # RRF 基线与 Qwen 精排整体对比

新增评测题时，只修改 evaluation/retrieval_cases.json；每题至少写一个
人工确认过的 expected.doc_title，可选写 section_contains 进一步精确到章节。
"""

import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List


# evaluation/ 的上一层是 code/；加入模块搜索路径后可直接导入项目模块。
CODE_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = CODE_DIR.parent
sys.path.insert(0, str(CODE_DIR))

from config import DEFAULT_CONFIG
from rag_modules import (
    DataPreparationModule,
    IndexConstructionModule,
    RetrievalOptimizationModule,
)


CASES_PATH = Path(__file__).with_name("retrieval_cases.json")


def load_cases(path: Path) -> List[Dict[str, Any]]:
    """读取评测题；结构不正确时尽早报出明确错误。"""
    with path.open("r", encoding="utf-8") as file:
        cases = json.load(file)

    if not isinstance(cases, list) or not cases:
        raise ValueError("评测集必须是至少包含一道题的 JSON 列表。")

    for case in cases:
        if not isinstance(case, dict) or not case.get("query") or not case.get("expected"):
            raise ValueError("每道题必须包含 query 和 expected 字段。")

    return cases


def is_expected_chunk(chunk, expected: Dict[str, str]) -> bool:
    """判断一个检索块是否符合人工标注的标准来源与可选章节。"""
    metadata = chunk.metadata if isinstance(chunk.metadata, dict) else {}

    if metadata.get("doc_title") != expected.get("doc_title"):
        return False

    section_contains = expected.get("section_contains")
    if section_contains and section_contains not in metadata.get("section", ""):
        return False

    return True


def evaluate_case(case: Dict[str, Any], retriever: RetrievalOptimizationModule, top_k: int) -> Dict[str, Any]:
    """评估单题，返回命中情况、首个相关结果排名和逐块明细。"""
    chunks = retriever.hybrid_search(case["query"], top_k=top_k)
    expected_sources = case["expected"]

    first_relevant_rank = None
    relevant_count = 0
    result_rows = []

    for rank, chunk in enumerate(chunks, 1):
        is_relevant = any(is_expected_chunk(chunk, expected) for expected in expected_sources)
        if is_relevant:
            relevant_count += 1
            if first_relevant_rank is None:
                first_relevant_rank = rank

        metadata = chunk.metadata if isinstance(chunk.metadata, dict) else {}
        result_rows.append({
            "rank": rank,
            "relevant": is_relevant,
            "doc_title": metadata.get("doc_title", "未知文档"),
            "section": metadata.get("section", "内容片段"),
            "rrf_score": metadata.get("_rrf_score"),
            "rerank_score": metadata.get("_rerank_score"),
        })

    return {
        "id": case.get("id", case["query"]),
        "query": case["query"],
        "hit": first_relevant_rank is not None,
        "first_relevant_rank": first_relevant_rank,
        "reciprocal_rank": 1 / first_relevant_rank if first_relevant_rank else 0.0,
        "precision": relevant_count / top_k,
        "rows": result_rows,
    }


def summarize(results: List[Dict[str, Any]]) -> Dict[str, float]:
    """将逐题结果汇总为可比较的检索指标。"""
    total = len(results)
    return {
        "hit_rate": sum(result["hit"] for result in results) / total,
        "mrr": sum(result["reciprocal_rank"] for result in results) / total,
        "precision": sum(result["precision"] for result in results) / total,
    }


def run_evaluation(
    label: str,
    cases: List[Dict[str, Any]],
    retriever: RetrievalOptimizationModule,
    top_k: int,
) -> tuple[List[Dict[str, Any]], float]:
    """执行一轮评测并返回逐题结果与纯检索耗时。"""
    print(f"\n{'=' * 70}\n开始评测：{label}")
    started_at = time.perf_counter()
    results = []
    for index, case in enumerate(cases, 1):
        print(f"[{label} {index}/{len(cases)}] 正在评测：{case['query']}", flush=True)
        case_started_at = time.perf_counter()
        results.append(evaluate_case(case, retriever, top_k))
        print(
            f"[{label} {index}/{len(cases)}] 完成，耗时 "
            f"{time.perf_counter() - case_started_at:.2f}s",
            flush=True,
        )
    return results, time.perf_counter() - started_at


def print_summary(label: str, results: List[Dict[str, Any]], top_k: int, elapsed: float):
    """打印单个方案的总指标。"""
    metrics = summarize(results)
    print(f"\n{label} 汇总：")
    print(f"  Hit@{top_k}：{metrics['hit_rate']:.2%}")
    print(f"  MRR：{metrics['mrr']:.4f}")
    print(f"  Precision@{top_k}：{metrics['precision']:.2%}")
    print(f"  纯检索评测耗时：{elapsed:.2f}s")
    return metrics


def print_comparison(
    baseline_results: List[Dict[str, Any]],
    rerank_results: List[Dict[str, Any]],
    top_k: int,
    baseline_elapsed: float,
    rerank_elapsed: float,
):
    """打印基线和精排方案的总体及逐题对比。"""
    baseline = print_summary("RRF 基线", baseline_results, top_k, baseline_elapsed)
    rerank = print_summary("RRF + Qwen rerank", rerank_results, top_k, rerank_elapsed)

    print(f"\n{'=' * 70}\n总体变化（精排 - 基线）：")
    print(f"  Hit@{top_k}：{rerank['hit_rate'] - baseline['hit_rate']:+.2%}")
    print(f"  MRR：{rerank['mrr'] - baseline['mrr']:+.4f}")
    print(f"  Precision@{top_k}：{rerank['precision'] - baseline['precision']:+.2%}")
    print(f"  耗时倍率：{rerank_elapsed / baseline_elapsed:.1f}x" if baseline_elapsed else "  耗时倍率：N/A")

    print("\n逐题首个相关结果排名（- 表示未命中）：")
    for base, ranked in zip(baseline_results, rerank_results):
        base_rank = base["first_relevant_rank"] or "-"
        rerank_rank = ranked["first_relevant_rank"] or "-"
        print(f"  {base['query']} | RRF: {base_rank} → Qwen: {rerank_rank}")


def make_retriever(
    vectorstore,
    chunks,
    config,
    rerank_enabled: bool,
) -> RetrievalOptimizationModule:
    """以同一数据和向量库构造不同检索策略，确保对比只改变是否启用精排。"""
    return RetrievalOptimizationModule(
        vectorstore,
        chunks,
        rerank_enabled=rerank_enabled,
        reranker_model=config.reranker_model,
        rerank_candidate_k=config.rerank_candidate_k,
        rerank_batch_size=config.rerank_batch_size,
        rerank_max_chars=config.rerank_max_chars,
        reranker_device=config.reranker_device,
    )


def main():
    """加载现有索引，运行单方案评测或基线与精排的整体对比。"""
    parser = argparse.ArgumentParser(description="评估 Medical RAG 检索效果。")
    parser.add_argument(
        "--no-rerank",
        action="store_true",
        help="关闭 Qwen 重排模型，仅评估 FAISS + BM25 + RRF 基线。",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="依次评估 RRF 基线与 RRF + Qwen rerank，并输出整体和逐题差异。",
    )
    args = parser.parse_args()
    if args.compare and args.no_rerank:
        parser.error("--compare 与 --no-rerank 不能同时使用。")

    config = replace(DEFAULT_CONFIG, rerank_enabled=not args.no_rerank)
    top_k = config.top_k
    cases = load_cases(CASES_PATH)

    mode = "RRF 基线 vs RRF + Qwen rerank" if args.compare else (
        "FAISS + BM25 + RRF + Qwen rerank" if config.rerank_enabled else "FAISS + BM25 + RRF"
    )
    print(f"读取评测题：{len(cases)} 道；模式：{mode}；评测指标：Hit@{top_k}、MRR、Precision@{top_k}")

    data_module = DataPreparationModule(str(PROJECT_ROOT / "data"))
    data_module.load_documents()
    chunks = data_module.chunk_documents()

    index_module = IndexConstructionModule(
        model_name=config.embedding_model,
        index_save_path=str(PROJECT_ROOT / "vector_index"),
    )
    vectorstore = index_module.load_index()
    if args.compare:
        baseline_retriever = make_retriever(vectorstore, chunks, config, rerank_enabled=False)
        rerank_retriever = make_retriever(vectorstore, chunks, config, rerank_enabled=True)
        baseline_results, baseline_elapsed = run_evaluation("RRF", cases, baseline_retriever, top_k)
        rerank_results, rerank_elapsed = run_evaluation("Qwen", cases, rerank_retriever, top_k)
        print_comparison(baseline_results, rerank_results, top_k, baseline_elapsed, rerank_elapsed)
    else:
        retriever = make_retriever(vectorstore, chunks, config, rerank_enabled=config.rerank_enabled)
        label = "Qwen" if config.rerank_enabled else "RRF"
        results, elapsed = run_evaluation(label, cases, retriever, top_k)
        print_summary(label, results, top_k, elapsed)

    print("说明：指标只代表当前已人工标注的题目；扩充 retrieval_cases.json 后再比较不同版本。")


if __name__ == "__main__":
    main()
