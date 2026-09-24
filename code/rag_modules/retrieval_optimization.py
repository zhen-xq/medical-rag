"""
检索优化模块 - 向量+BM25双路检索与RRF融合排序
"""

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
    VECTOR_RETRIEVER_K = 15   # FAISS 向量检索单独先拿 top 15 候选
    BM25_RETRIEVER_K   = 15   # BM25 关键词检索单独先拿 top 15 候选
    RRF_K              = 60   # RRF 公式平滑系数 k（参考实现同样取 60，经验值）
    FILTER_MULTIPLIER  = 3    # metadata_filtered_search 先拿 top_k*3 多候选再 Python 侧过滤，避免直接被筛空

    def __init__(self, vectorstore: FAISS, chunks: List[Document]):
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

        # =============================================================
        # ③ 立刻调用 setup_retrievers（跟参考实现一模一样：__init__ 里直接完成初始化，
        #    避免后续用户忘了调 setup 直接 hybrid_search 崩）
        # =============================================================
        self.setup_retrievers()

    def setup_retrievers(self):
        """
        设置向量检索器和 BM25 检索器（完全仿照参考实现，只是把 k 从 5 改成 15 跟架构文档一致）
        结果写进成员变量：
          - self.vector_retriever：FAISS 的 similarity 向量检索器，k=15
          - self.bm25_retriever：BM25Retriever（基于 self.chunks 建的关键词倒排索引），k=15
        """
        # =============================================================
        # ① 向量检索器：用 self.vectorstore.as_retriever(...)，search_type="similarity" + search_kwargs k=15
        #    说明：as_retriever() 包装后可以统一 invoke(query) 接口，后面 hybrid_search 两个检索器都用 .invoke() 调用即可（跟参考一样）
        # =============================================================
        self.vector_retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": self.VECTOR_RETRIEVER_K},
        )

        # =============================================================
        # ② BM25 关键词检索器：BM25Retriever.from_documents(self.chunks, k=15)
        #    说明：from_documents 会对所有 chunk 的 page_content 做中文分词（内部默认是 jieba-like 的字符 n-gram + 通用 tokenizer，
        #    对中文还行，后续你要升级成专门中文分词（jieba）再换；当前完全照着参考实现来，不引入额外依赖）
        # =============================================================
        try:
            self.bm25_retriever = BM25Retriever.from_documents(
                self.chunks,
                k=self.BM25_RETRIEVER_K,
            )
        except ImportError as e:
            print(
                "[ERROR][RetrievalOptimizationModule.setup_retrievers] 缺少 rank_bm25 依赖（BM25Retriever 需要它）。\n"
                "  请先执行：\n"
                "    pip install rank_bm25\n"
                f"  原始 ImportError：{e}"
            )
            raise

    def hybrid_search(self, query: str, top_k: int = 3) -> List[Document]:
        """
        混合检索 - 结合向量检索（语义相似）和 BM25 检索（字面关键词相似），使用 RRF 融合重排，最后切 top_k
        完全仿照参考实现的 3 步流：vector_retriever.invoke → bm25_retriever.invoke → _rrf_rerank → 切片。

        Args:
            query: 用户的原始查询文本（如 "青光眼有什么症状？"）
            top_k: 最终返回结果数量（架构文档约定默认 top_k=3）

        Returns:
            List[Document] —— RRF 重排后的前 top_k 个 chunk，每个 doc.metadata["_rrf_score"] 里带 RRF 融合分数
        """
        # =============================================================
        # ① 分别跑向量检索和 BM25 检索（两个独立，将来甚至可以线程池并行，这里先按参考顺序跑）
        # =============================================================
        vector_docs = self.vector_retriever.invoke(query)
        bm25_docs   = self.bm25_retriever.invoke(query)

        # =============================================================
        # ② RRF 融合重排
        # =============================================================
        reranked_docs = self._rrf_rerank(vector_docs, bm25_docs)

        # =============================================================
        # ③ 切 top_k 返回（不够就返回多少，不要补空/崩）
        # =============================================================
        return reranked_docs[:top_k]

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
        # ② 先 hybrid_search 拿 FILTER_MULTIPLIER * top_k 多条候选（默认 3 倍 = 15 条）再过滤
        #    （如果先 filter 再搜的话 FAISS 内部 filter 不同版本坑多，Python 侧过滤简单稳，跟参考一致）
        # =============================================================
        candidates = self.hybrid_search(query, top_k=top_k * self.FILTER_MULTIPLIER)

        # =============================================================
        # ③ Python 侧逐个 candidate 应用 filters（多 key and 语义；value 为 list 则 any 命中即算匹配）
        # =============================================================
        filtered_docs = []
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
                # --- 提前终止：已经凑够 top_k 条就不用再遍历了（省一点点时间）---
                if len(filtered_docs) >= top_k:
                    break

        return filtered_docs

    def _rrf_rerank(self, vector_docs: List[Document], bm25_docs: List[Document], k: int = RRF_K) -> List[Document]:
        """
        使用 RRF (Reciprocal Rank Fusion) 算法重排文档（完全仿照参考实现公式：RRF_score = Σ 1 / (k + rank + 1)）
        唯一的区别就是「唯一 doc_id 的计算方式」：
            - 参考：hashlib.md5(doc.page_content.encode()).hexdigest()（基于内容哈希）
            - 我们：直接用 doc.metadata["chunk_id"]（全局唯一，不会因为内容相同但来源不同的 chunk 被误去重）
        RRF 分数写到每个返回 doc.metadata["_rrf_score"] 里，方便后面调试/打印/做二次重排。

        Args:
            vector_docs: FAISS 向量检索结果（长度 VECTOR_RETRIEVER_K=15）
            bm25_docs:   BM25 关键词检索结果（长度 BM25_RETRIEVER_K=15）
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
