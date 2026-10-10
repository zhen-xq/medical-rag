# 03 索引构建 + 检索优化

## A. IndexConstructionModule
**文件**: `code/rag_modules/index_construction.py`

### 类成员
```python
self.model_name = "BAAI/bge-small-zh-v1.5"
self.index_save_path: str
self.embeddings: HuggingFaceEmbeddings | None
self.vectorstore: FAISS | None
```

---

### 1. setup_embeddings()

```python
from langchain_huggingface import HuggingFaceEmbeddings

self.embeddings = HuggingFaceEmbeddings(
    model_name=self.model_name,
    model_kwargs={'device': 'cpu'},       # 有GPU改 'cuda'
    encode_kwargs={'normalize_embeddings': True}
)
```
**说明**: __init__ 中立刻调用。

---

### 2. build_vector_index(chunks: List[Document]) -> FAISS

```python
from langchain_community.vectorstores import FAISS

texts = [c.page_content for c in chunks]
metadatas = [c.metadata for c in chunks]
self.vectorstore = FAISS.from_texts(texts=texts, embedding=self.embeddings, metadatas=metadatas)
return self.vectorstore
```

---

### 3. save_index()

前置: `self.vectorstore` 非空

```python
Path(self.index_save_path).mkdir(parents=True, exist_ok=True)
self.vectorstore.save_local(self.index_save_path)
```
生成: `index.faiss` + `index.pkl`

---

### 4. load_index() -> FAISS | None

```python
if not self.embeddings:
    self.setup_embeddings()
if not Path(self.index_save_path).exists():
    return None
self.vectorstore = FAISS.load_local(
    self.index_save_path,
    self.embeddings,
    allow_dangerous_deserialization=True
)
return self.vectorstore
```

---

## B. RetrievalOptimizationModule
**文件**: `code/rag_modules/retrieval_optimization.py`

**当前检索策略**：FAISS 语义检索与 BM25 关键词检索各自召回 30 个候选，随后使用
RRF（Reciprocal Rank Fusion）去重并融合排序；默认再由 Qwen CrossEncoder 模型对 RRF
候选做语义精排。

### 类成员
```python
self.vectorstore: FAISS
self.chunks: List[Document]
self.vector_retriever            # FAISS.as_retriever()
self.bm25_retriever              # BM25Retriever.from_documents()
```

---

### 1. setup_retrievers()

```python
from langchain_community.retrievers import BM25Retriever

self.vector_retriever = self.vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 30}       # RRF 前多召回一些，供精排使用
)

self.bm25_retriever = BM25Retriever.from_documents(
    self.chunks,
    k=30,
    preprocess_func=self._chinese_bm25_tokenize,
)
```
**说明**:

- 在 `__init__` 中立刻调用。
- BM25 使用 `_chinese_bm25_tokenize` 进行中文分词：连续中文会产生“单字 + 相邻双字词”两类 token；英文、数字和缩写会作为完整 token 保留。因此它能处理中文术语、`OCT`、`IOL`、视力数值等关键词，避免默认 `text.split()` 在中文文本上退化。

---

### 2. hybrid_search(query: str, top_k: int = 3) -> List[Document]

```
1. vector_docs = self.vector_retriever.invoke(query)   # 30条
2. bm25_docs   = self.bm25_retriever.invoke(query)     # 30条
3. reranked = self._rrf_rerank(vector_docs, bm25_docs)                 # 去重+融合排序
4. final_docs = self.rerank(query, reranked[:20], top_k)               # Qwen 精排（默认开启）
5. return final_docs
```

返回的每个文档会在 `metadata["_rrf_score"]` 中保留融合得分；启用模型精排时，还会保留 `_rerank_score`。

---

### 3. _rrf_rerank(vector_results, bm25_results) -> List[Document]

```python
RRF_K = 60
doc_scores = {}         # chunk_id -> score
doc_objects = {}        # chunk_id -> Document

# 向量检索计分
for rank, doc in enumerate(vector_results):
    doc_id = doc.metadata.get("chunk_id") or md5(doc.page_content)
    doc_objects[doc_id] = doc
    doc_scores[doc_id] = doc_scores.get(doc_id, 0) + 1 / (RRF_K + rank + 1)

# BM25计分
for rank, doc in enumerate(bm25_results):
    doc_id = doc.metadata.get("chunk_id") or md5(doc.page_content)
    doc_objects[doc_id] = doc
    doc_scores[doc_id] = doc_scores.get(doc_id, 0) + 1 / (RRF_K + rank + 1)

# 按融合分数降序、chunk_id 升序（保证同分时稳定可复现）
sorted_items = sorted(doc_scores.items(), key=lambda x: (-x[1], x[0]))
reranked_docs = []
for doc_id, score in sorted_items:
    doc = doc_objects[doc_id]
    new_meta = dict(doc.metadata)
    new_meta["_rrf_score"] = round(score, 8)
    reranked_docs.append(Document(page_content=doc.page_content, metadata=new_meta))
return reranked_docs
```

公式为 `RRF_score = Σ 1 / (60 + rank + 1)`。同一个切块被两路同时命中时分数会叠加；去重优先使用稳定的 `chunk_id`，缺失时才回退到正文 MD5，从而避免用 Python 对象地址去重导致跨检索器无法匹配。

---

### 4. metadata_filtered_search(query, filters: Dict, top_k=5) -> List[Document]

```python
# 先做混合召回和 RRF，再在 Python 侧过滤，最后才调用模型精排。
candidates = self._hybrid_retrieve(query)
results = [doc for doc in candidates if metadata_matches(doc.metadata, filters)]
return self._finalize_results(query, results, top_k)
```

**当前行为与语义**:

- 过滤发生在 RRF 融合之后，不直接使用 FAISS 的内部 `filter` 参数；这样可保持 FAISS 与 BM25 两路都参与召回。
- 过滤会检查完整的 RRF 融合池；随后才从前 20 个过滤后候选中选择精排输入。
- 多个字段之间为 AND；单个字段值为列表时采用 OR；缺少指定 metadata 的文档不匹配。
- 可用字段包括 `category`、`sub_category`、`source_level`、`doc_type`、`difficulty`。例如：`{"category": "diseases", "sub_category": "ophthalmology"}`、`{"source_level": "A"}`。
- 由于过滤在召回之后，严格条件下可能出现“全库前列候选均不匹配而返回空结果”的情况；这是当前实现的已知边界，而不是 FAISS 原生过滤。

---

## C. Qwen CrossEncoder 精排

默认模型为 `Qwen/Qwen3-Reranker-0.6B`。该模型不会替代向量模型，而是在候选已经被快速缩小后，将每一个 `(query, chunk)` 对一起编码并预测相关性分数。首次检索到达精排阶段时才加载模型；因此系统建库、仅使用基线评测或关闭 `rerank_enabled` 时不会加载它。

```python
reranker = CrossEncoder("Qwen/Qwen3-Reranker-0.6B", trust_remote_code=True)
pairs = [(query, doc.page_content[:1500]) for doc in candidates[:20]]
scores = reranker.predict(pairs, batch_size=4)
```

精排后的每个文档保留 `_rerank_score`。可通过 `MedicalRAGConfig` 调整：`rerank_enabled`、`reranker_model`、`rerank_candidate_k`、`rerank_batch_size`、`rerank_max_chars` 与 `reranker_device`。未指定设备时，代码会优先使用 CUDA，否则回退 CPU。评测脚本默认评估精排流程；添加 `--no-rerank` 可得到 RRF 基线，便于比较 Hit@K、MRR 和 Precision@K。

---

## D. 与主问答流程的衔接

`main.py` 在检索前先调用查询路由：`list` 类型保持原问题，`detail` 和 `general` 类型会经 LLM 查询改写后再检索。随后根据用户问题中出现的分类或难度关键词，自动选择普通混合检索或元数据过滤检索。

最终默认仅将模型精排后的前 3 个子块交给生成模块；关闭精排时则使用 RRF 前 3 个子块。资料采用“Markdown 标题优先 + 递归字符切分”的方式形成子块：标题保留在正文及 `section` metadata 中；长度超过 1500 字符的标题段会以 200 字符重叠继续切分。
