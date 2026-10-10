"""
检索优化模块 - 向量+BM25双路检索与RRF融合排序
"""

import re
import warnings
from typing import List, Dict, Any

from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

# 忽略 BM25Retriever.from_documents 的 DeprecationWarning
warnings.filterwarnings("ignore", message=".*deprecated.*", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*deprecated.*", category=FutureWarning)


class RetrievalOptimizationModule:
    """检索优化模块 - 负责混合检索（向量 + BM25 关键词）+ RRF 融合重排 + metadata 条件过滤"""

    # =============================================================
    # 检索阶段的常量化配置（与 03_index_retrieval.md 文档一致）
    # =============================================================
    VECTOR_RETRIEVER_K = 30   # FAISS 向量检索单独先拿 top 30 候选，供后续精排
    BM25_RETRIEVER_K   = 30   # BM25 关键词检索单独先拿 top 30 候选，供后续精排
    RRF_K              = 60   # RRF 公式平滑系数 k（参考实现同样取 60，经验值）

    @staticmethod
    def _chinese_bm25_tokenize(text: str) -> List[str]:
        """
        为 BM25 提供无第三方依赖的中文分词。

        LangChain 默认使用 text.split()，只适合以空格分词的语言；中文资料没有空格时，
        一整段内容会退化成一个 token，查询几乎无法产生关键词匹配。

        这里对连续中文生成「单字 + 双字词组」两类 token：
          "青光眼常见症状" → ["青", "光", "眼", ..., "青光", "光眼", "眼常", ...]
        单字保证短术语也能召回，双字词组让“青光眼”“斜视”“弱视”等连续术语有更高区分度。
        英文、数字和缩写保留为 token，适用于 OCT、IOL、视力数值等检索词。
        """
        if not text:
            return []

        segments = re.findall(r"[\u4e00-\u9fff]+|[a-zA-Z0-9]+", text.lower())
        tokens: List[str] = []

        for segment in segments:
            if re.fullmatch(r"[\u4e00-\u9fff]+", segment):
                chars = list(segment)
                tokens.extend(chars)
                tokens.extend(
                    segment[index:index + 2]
                    for index in range(len(segment) - 1)
                )
            else:
                tokens.append(segment)

        return tokens

    def __init__(
        self,
        vectorstore: FAISS,
        chunks: List[Document],
        rerank_enabled: bool = True,
        reranker_model: str = "Qwen/Qwen3-Reranker-0.6B",
        rerank_candidate_k: int = 20,
        rerank_batch_size: int = 8,
        rerank_max_chars: int = 1500,
        reranker_device: str | None = None,
    ):
        """
        初始化检索优化模块（完全仿照参考：保存 vectorstore 和 chunks，然后立刻调用 setup_retrievers）

        Args:
            vectorstore: 已构建完成的 FAISS 向量存储（来自 IndexConstructionModule.vectorstore / load_index）
            chunks: 全量子块列表（来自 DataPreparationModule.chunk_documents()，BM25Retriever 要拿这个建倒排表）
        """
        # =============================================================
        # ① 先做入参防御：不写死断言崩溃，而是给中文提示再 raise（调错了也知道哪里错）
        # =============================================================
        if vectorstore is None:
            raise ValueError(
                "[ERROR][RetrievalOptimizationModule.__init__] vectorstore 不能是 None！"
                " 请先执行 IndexConstructionModule.build_vector_index(chunks) 或 load_index()，再把返回的 vectorstore 传进来。"
            )
        if not chunks:
            raise ValueError(
                "[ERROR][RetrievalOptimizationModule.__init__] chunks 为空列表！"
                " BM25 检索器需要 chunk 列表建倒排索引。请先执行 DataPreparationModule.chunk_documents() 再传进来。"
            )

        # =============================================================
        # ② 保存成员变量（完全仿照参考：self.vectorstore / self.chunks）
        # =============================================================
        self.vectorstore = vectorstore
        self.chunks = chunks
        self.rerank_enabled = rerank_enabled
        self.reranker_model_name = reranker_model
        self.rerank_candidate_k = max(1, rerank_candidate_k)
        self.rerank_batch_size = max(1, rerank_batch_size)
        self.rerank_max_chars = max(1, rerank_max_chars)
        self.reranker_device = reranker_device
        self._reranker = None

        # =============================================================
        # ③ 立刻调用 setup_retrievers（跟参考实现一模一样：__init__ 里直接完成初始化，
        #    避免后续用户忘了调 setup 直接 hybrid_search 崩）
        # =============================================================
        self.setup_retrievers()

    def _load_reranker(self):
        """按需加载 CrossEncoder，避免仅建库或只做基线评测时下载/占用重排模型。"""
        if self._reranker is not None:
            return self._reranker

        try:
            from sentence_transformers import CrossEncoder
        except ImportError as e:
            raise ImportError(
                "启用了 rerank，但缺少 sentence-transformers。请执行 pip install -r requirements.txt。"
            ) from e

        resolved_device = self._resolve_reranker_device()
        print(
            f"[RetrievalOptimization] 正在加载重排模型：{self.reranker_model_name} "
            f"(device={resolved_device}, batch_size={self.rerank_batch_size})"
        )
        try:
            self._reranker = CrossEncoder(
                self.reranker_model_name,
                device=resolved_device,
                trust_remote_code=True,
            )
        except Exception as e:
            raise RuntimeError(
                f"重排模型加载失败：{self.reranker_model_name}。"
                "请检查本地模型缓存、网络访问、transformers/sentence-transformers 版本及可用显存。"
            ) from e

        print("[RetrievalOptimization] 重排模型加载完成")
        return self._reranker

    def _resolve_reranker_device(self) -> str:
        """优先使用显式设备；未配置时自动选择 CUDA，无法使用时回退 CPU。"""
        if self.reranker_device:
            return self.reranker_device
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"

    def setup_retrievers(self):
        """
        设置向量检索器和 BM25 检索器；两路各保留 30 条候选，供 RRF 与模型精排使用。
        结果写进成员变量：
          - self.vector_retriever：FAISS 的 similarity 向量检索器，k=30
          - self.bm25_retriever：BM25Retriever（基于 self.chunks 建的关键词倒排索引），k=30
        """
        # =============================================================
        # ① 向量检索器：用 self.vectorstore.as_retriever(...)，search_type="similarity" + search_kwargs k=30
        #    说明：as_retriever() 包装后可以统一 invoke(query) 接口，后面 hybrid_search 两个检索器都用 .invoke() 调用即可（跟参考一样）
        # =============================================================
        self.vector_retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": self.VECTOR_RETRIEVER_K},
        )

        # =============================================================
        # ② BM25 关键词检索器：显式传入中文 tokenizer。
        #    LangChain 默认仅 text.split()，中文无空格会导致整段文本成为单一 token，
        #    因此必须使用 _chinese_bm25_tokenize 才能让关键词检索真正参与 RRF 融合。
        # =============================================================
        try:
            self.bm25_retriever = BM25Retriever.from_documents(
                self.chunks,
                k=self.BM25_RETRIEVER_K,
                preprocess_func=self._chinese_bm25_tokenize,
            )
        except ImportError as e:
            print(
                "[ERROR][RetrievalOptimizationModule.setup_retrievers] 缺少 rank_bm25 依赖（BM25Retriever 需要它）。\n"
                "  请先执行：\n"
                "    pip install rank_bm25\n"
                f"  原始 ImportError：{e}"
            )
            raise

    def _hybrid_retrieve(self, query: str) -> List[Document]:
        """执行双路召回和 RRF，只返回完整融合候选池，不在此处截断或调用重排模型。"""
        vector_docs = self.vector_retriever.invoke(query)
        bm25_docs = self.bm25_retriever.invoke(query)
        return self._rrf_rerank(vector_docs, bm25_docs)

    def rerank(self, query: str, candidates: List[Document], top_k: int) -> List[Document]:
        """以 CrossEncoder 对 query-document 对逐对评分，返回模型相关性最高的 top_k。"""
        if not candidates or top_k <= 0:
            return []

        limited_candidates = candidates[:self.rerank_candidate_k]
        reranker = self._load_reranker()
        pairs = [
            (query, doc.page_content[:self.rerank_max_chars])
            for doc in limited_candidates
        ]
        try:
            scores = reranker.predict(pairs, batch_size=self.rerank_batch_size)
        except Exception as e:
            raise RuntimeError("重排模型推理失败，请检查设备内存和 rerank_batch_size 配置。") from e

        ranked = sorted(
            zip(limited_candidates, scores),
            key=lambda item: float(item[1]),
            reverse=True,
        )
        reranked_docs = []
        for doc, score in ranked[:top_k]:
            new_meta = dict(doc.metadata)
            new_meta["_rerank_score"] = round(float(score), 8)
            try:
                new_doc = doc.copy()
                new_doc.metadata = new_meta
            except Exception:
                new_doc = Document(page_content=doc.page_content, metadata=new_meta)
            reranked_docs.append(new_doc)
        return reranked_docs

    def _finalize_results(self, query: str, candidates: List[Document], top_k: int) -> List[Document]:
        """根据配置选择 RRF 直出或 RRF 后模型精排。"""
        if self.rerank_enabled:
            return self.rerank(query, candidates, top_k)
        return candidates[:top_k]

    def hybrid_search(self, query: str, top_k: int = 3) -> List[Document]:
        """
        混合检索 - 结合向量检索、BM25、RRF 融合，并可选调用 CrossEncoder 精排。

        Args:
            query: 用户的原始查询文本（如 "青光眼有什么症状？"）
            top_k: 最终返回结果数量（架构文档约定默认 top_k=3）

        Returns:
            List[Document] —— 最终前 top_k 个 chunk；始终带 _rrf_score，启用精排时额外带 _rerank_score
        """
        candidates = self._hybrid_retrieve(query)
        return self._finalize_results(query, candidates, top_k)

    def metadata_filtered_search(self, query: str, filters: Dict[str, Any], top_k: int = 5) -> List[Document]:
        """
        带元数据过滤的检索（完全仿照参考：先拿更多候选再 Python 侧硬过滤，不依赖 FAISS 内部的 filter 实现（兼容性差））

        支持两种 filter 值写法（跟参考完全一致）：
          a. 单值：  filters = {"category": "diseases"}                → metadata["category"] 必须 == "diseases"
          b. 列表：  filters = {"sub_category": ["ophthalmology", "optometry"]}  → 值只要在列表里就算命中（or 语义）
        多个 key 之间是 and 语义：所有 key 都匹配才算命中。

        我们 medical RAG 的常用 filters 示例（字段名就是 DataPreparation 里写的那些，直接拿这些用）：
            {"category": "diseases", "sub_category": "ophthalmology"}                         → 只在眼科临床教材里搜
            {"source_level": "A"}                                                               → 只搜 Wills/眼科学第十版 这种 A 级权威资料
            {"doc_type": ["vocational", "notes"], "category": "health_articles"}                → 只搜职业资格教材 + 门诊笔记

        Args:
            query:   查询文本
            filters: 元数据过滤条件 dict；空 dict 就等价于不做过滤，直接 hybrid_search 返回 top_k
            top_k:   最终返回结果数量
        Returns:
            List[Document] —— 过滤后的文档列表（最多 top_k 条；可能不足 top_k，如果过滤太严没命中就返回 []）
        """
        # =============================================================
        # ① 入参防御：filters 为空 → 直接 hybrid_search 返回即可，不浪费计算
        # =============================================================
        if not filters:
            return self.hybrid_search(query, top_k=top_k)

        # =============================================================
        # ② 先获得完整 RRF 融合池，再 Python 侧过滤；这样严格过滤不会受最终 top_k 截断影响。
        # =============================================================
        # 先在完整 RRF 候选池中筛选，再调用重排模型；这样重排不会浪费在最终会被过滤掉的片段上。
        candidates = self._hybrid_retrieve(query)

        # =============================================================
        # ③ Python 侧逐个 candidate 应用 filters（多 key and 语义；value 为 list 则 any 命中即算匹配）
        # =============================================================
        filtered_docs = []
        candidate_limit = max(top_k, self.rerank_candidate_k) if self.rerank_enabled else top_k
        for doc in candidates:
            match = True
            for key, value in filters.items():
                # --- 如果 metadata 里根本没有这个 key → 直接算不匹配（安全策略：不匹配默认丢，不蒙混）---
                if key not in doc.metadata:
                    match = False
                    break
                actual = doc.metadata[key]
                # --- value 是 list → 实际值在 list 里就算命中（or 语义）---
                if isinstance(value, list):
                    if actual not in value:
                        match = False
                        break
                # --- value 不是 list → 严格相等（不做类型强转，避免 "14"==14 的乌龙）---
                else:
                    if actual != value:
                        match = False
                        break

            if match:
                filtered_docs.append(doc)
                # 启用精排时保留足够候选，避免模型只能在最终 top_k 条中重新排序。
                if len(filtered_docs) >= candidate_limit:
                    break

        return self._finalize_results(query, filtered_docs, top_k)

    def _rrf_rerank(self, vector_docs: List[Document], bm25_docs: List[Document], k: int = RRF_K) -> List[Document]:
        """
        使用 RRF (Reciprocal Rank Fusion) 算法重排文档（完全仿照参考实现公式：RRF_score = Σ 1 / (k + rank + 1)）
        唯一的区别就是「唯一 doc_id 的计算方式」：
            - 参考：hashlib.md5(doc.page_content.encode()).hexdigest()（基于内容哈希）
            - 我们：直接用 doc.metadata["chunk_id"]（全局唯一，不会因为内容相同但来源不同的 chunk 被误去重）
        RRF 分数写到每个返回 doc.metadata["_rrf_score"] 里，方便后面调试/打印/做二次重排。

        Args:
            vector_docs: FAISS 向量检索结果（长度 VECTOR_RETRIEVER_K=30）
            bm25_docs:   BM25 关键词检索结果（长度 BM25_RETRIEVER_K=30）
            k:           RRF 平滑系数（默认 60，和参考一致）

        Returns:
            List[Document] —— 按总 RRF 分数从高到低排序后的去重合并列表（每个 doc.metadata 含 _rrf_score）
        """
        # =============================================================
        # 两个 dict：分数累加器 + 实际 Document 对象保存（最后按分数把对象捞出来）
        # =============================================================
        doc_scores  = {}   # {chunk_id: 累加后的总 RRF 分数}
        doc_objects = {}   # {chunk_id: 原始 Document 对象（如果同一份 chunk 被两个检索器同时命中，保留后面那个不影响，反正内容相同）}

        # =============================================================
        # ① 累加向量检索结果的 RRF 分数（rank 从 0 开始，所以公式 1/(k + rank + 1)）
        # =============================================================
        for rank, doc in enumerate(vector_docs):
            doc_id = self._get_doc_unique_id(doc)                # 用 chunk_id 做 key（没有 chunk_id 的脏数据再回退 md5 page_content）
            doc_objects[doc_id] = doc
            # 参考实现完全一样的公式：1.0 / (k + rank + 1)
            rrf_score = 1.0 / (k + rank + 1)
            doc_scores[doc_id] = doc_scores.get(doc_id, 0.0) + rrf_score

        # =============================================================
        # ② 累加 BM25 检索结果的 RRF 分数（同一份 chunk 被两边同时命中的话分数就叠加，排更前面）
        # =============================================================
        for rank, doc in enumerate(bm25_docs):
            doc_id = self._get_doc_unique_id(doc)
            doc_objects[doc_id] = doc
            rrf_score = 1.0 / (k + rank + 1)
            doc_scores[doc_id] = doc_scores.get(doc_id, 0.0) + rrf_score

        # =============================================================
        # ③ 按最终 RRF 分数**从高到低**排序（reverse=True）
        #    sort 的 key 是 (总分, doc_id) —— 加一个 doc_id 的 tie-breaker 保证总分相同的情况下排序是稳定可复现的
        # =============================================================
        sorted_items = sorted(
            doc_scores.items(),
            key=lambda kv: (-kv[1], kv[0]),   # 负号表示按分数降序；然后 doc_id 升序 tie-breaker
        )

        # =============================================================
        # ④ 构建最终结果：把 doc_objects 里的 Document 拿出来，metadata 里塞进去 _rrf_score（临时字段），并保持排序一致
        # =============================================================
        reranked_docs = []
        for doc_id, final_score in sorted_items:
            if doc_id not in doc_objects:
                continue
            doc = doc_objects[doc_id]
            # --- 注意：不直接修改原 doc.metadata（会污染 self.chunks / self.vectorstore 里的原始对象），
            #     跟 DataPreparation 的 get_parent_documents 一样，先做 dict 拷贝再塞 _rrf_score ---
            new_meta = dict(doc.metadata)
            new_meta["_rrf_score"] = round(final_score, 8)   # 保留 8 位小数够精度了，打印好看
            try:
                new_doc = doc.copy()
                new_doc.metadata = new_meta
            except Exception:
                # 老版本 LangChain Document 可能没有 copy() 方法的兜底：直接 new 一个
                new_doc = Document(page_content=doc.page_content, metadata=new_meta)
            reranked_docs.append(new_doc)

        return reranked_docs

    @staticmethod
    def _get_doc_unique_id(doc: Document) -> str:
        """
        RRF 阶段用的「chunk 唯一 ID」。
        优先顺序：
          1) doc.metadata["chunk_id"]（我们 DataPreparation 每个 chunk 都写了，全局唯一，首选）
          2) 如果是脏数据（缺 chunk_id，比如有人手工 new Document 传进来）→ 回退到参考实现的 md5(page_content)，保证总有值不崩
        """
        mid = doc.metadata.get("chunk_id") if isinstance(doc.metadata, dict) else None
        if mid:
            return str(mid)
        # 回退：内容 md5（参考实现的老做法）
        import hashlib
        return hashlib.md5((doc.page_content or "").encode("utf-8")).hexdigest()
