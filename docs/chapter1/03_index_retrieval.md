# 03 索引构建 + 检索优化

## A. IndexConstructionModule
**文件**: `code/medical_rag/rag_modules/index_construction.py`

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
**文件**: `code/medical_rag/rag_modules/retrieval_optimization.py`

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
    search_kwargs={"k": 15}       # RRF前多召回一些
)

self.bm25_retriever = BM25Retriever.from_documents(
    documents=self.chunks,
    k=15
)
```
**说明**: __init__ 中立刻调用。

---

### 2. hybrid_search(query: str, top_k: int = 3) -> List[Document]

```
1. vector_docs = self.vector_retriever.get_relevant_documents(query)   # 15条
2. bm25_docs   = self.bm25_retriever.get_relevant_documents(query)     # 15条
3. reranked = self._rrf_rerank(vector_docs, bm25_docs)                 # 去重+融合排序
4. return reranked[:top_k]
```

---

### 3. _rrf_rerank(vector_results, bm25_results) -> List[Document]

```python
RRF_K = 60
rrf_scores = {}        # doc_id -> score

# 向量检索计分
for rank, doc in enumerate(vector_results):
    doc_id = id(doc)
    rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + 1 / (RRF_K + rank + 1)

# BM25计分
for rank, doc in enumerate(bm25_results):
    doc_id = id(doc)
    rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + 1 / (RRF_K + rank + 1)

# 去重池
all_docs = {id(doc): doc for doc in vector_results + bm25_results}

# 按score降序
sorted_items = sorted(all_docs.items(), key=lambda x: rrf_scores.get(x[0], 0), reverse=True)
return [doc for _, doc in sorted_items]
```

---

### 4. metadata_filtered_search(query, filters: Dict, top_k=5) -> List[Document]

```python
filtered_retriever = self.vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": top_k * 3, "filter": filters}   # 过滤后池小，放大3倍
)
results = filtered_retriever.invoke(query)
return results[:top_k]
```
**filters示例**: `{"category": "drug", "sub_category": "降压药"}`
