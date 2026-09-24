# 02 数据准备模块 DataPreparationModule

**文件**: `code/medical_rag/rag_modules/data_preparation.py`

## 类成员
```python
self.data_path: str              # 输入
self.documents: List[Document]   # 父文档列表（输出）
self.chunks: List[Document]      # 子块列表（输出）
self.parent_child_map: Dict[str, str]  # chunk_id -> parent_id（输出）
```

---

## 1. load_documents() -> List[Document]

**步骤**:
1. 递归 `Path(self.data_path).rglob("*.md")`（先只做md，pdf后续扩展）
2. 每个文件：
   - `open(f, 'r', encoding='utf-8').read()` 得 content
   - `parent_id = str(uuid.uuid4())`
   - 创建 `Document(page_content=content, metadata={"source": str(f), "parent_id": parent_id, "doc_type": "parent"})`
3. 对每个 doc 调 `_enhance_metadata(doc)`
4. `self.documents = documents`，返回

---

## 2. _enhance_metadata(doc: Document)

**从文件路径推断**（`Path(doc.metadata['source'])` 的 parts）：

| 路径含 | category | sub_category |
|-------|----------|-------------|
| `/diseases/cardiovascular/` | disease | 心血管内科 |
| `/diseases/respiratory/` | disease | 呼吸内科 |
| `/diseases/ophthalmology/` | disease | 眼科 |
| `/drugs/antibiotics/` | drug | 抗生素 |
| `/drugs/antihypertensives/` | drug | 降压药 |
| `/guidelines/` | guideline | 临床指南 |
| `/health_articles/` | article | 健康科普 |
| 其他 | other | 未分类 |

**source_level**:
- guideline → A
- drug / disease → B
- article → C

**doc_title**:
- 读 page_content 第一个 `# xxx` 行，去 `#` 和空格
- 找不到就用 `Path(source).stem`（文件名去扩展名）

**severity**（可选）:
- doc_title 含 心梗/脑出血/癌/衰竭 → critical
- 其他 → moderate

---

## 3. chunk_documents() -> List[Document]

**前置**: `self.documents` 非空

**步骤**:
1. 调 `_markdown_header_split()` → 得初步 chunks
2. 遍历 chunks，若 `len(chunk.page_content) > 1500`：
   - 用 `RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_documents([chunk])` 二次切
   - 替换原子块为切分结果，每个新 chunk 继承元数据，重新生成 `chunk_id`
3. 为最终每个 chunk 加：
   - `metadata['batch_index'] = 序号`
   - `metadata['chunk_size'] = len(chunk.page_content)`
4. `self.chunks = chunks`，返回

---

## 4. _markdown_header_split() -> List[Document]

**工具**: `from langchain_text_splitters import MarkdownHeaderTextSplitter`

**配置**:
```python
headers_to_split_on = [("#", "Header 1"), ("##", "Header 2"), ("###", "Header 3")]
splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on, strip_headers=False)
```

**对每个父文档 doc**:
1. `parent_id = doc.metadata['parent_id']`
2. `md_chunks = splitter.split_text(doc.page_content)`
3. 对 `i, chunk in enumerate(md_chunks)`：
   - `child_id = str(uuid.uuid4())`
   - `chunk.metadata.update(doc.metadata)` 先继承父元数据
   - `chunk.metadata.update({"chunk_id": child_id, "parent_id": parent_id, "doc_type": "child", "chunk_index": i})`
   - `self.parent_child_map[child_id] = parent_id`
4. `all_chunks.extend(md_chunks)`

返回 all_chunks

---

## 5. get_parent_documents(child_chunks: List[Document]) -> List[Document]

**步骤**:
1. `parent_relevance = {}` 存 `parent_id -> 命中次数`
2. `parent_docs_map = {}` 存 `parent_id -> Document`（缓存）
3. 遍历 child_chunks：
   - `parent_id = chunk.metadata.get('parent_id')`
   - `parent_relevance[parent_id] += 1`
   - 若 parent_id 不在 map，在 `self.documents` 中找匹配 doc，入 map
4. `sorted_ids = sorted(parent_relevance.keys(), key=lambda x: parent_relevance[x], reverse=True)`
5. 返回 `[parent_docs_map[id] for id in sorted_ids if id in parent_docs_map]`

---

## 6. get_statistics() -> Dict

返回：
```python
{
    "total_documents": len(self.documents),
    "total_chunks": len(self.chunks),
    "categories": Counter(d.metadata.get('category') for d in self.documents),
    "source_levels": Counter(d.metadata.get('source_level') for d in self.documents),
    "sub_categories": Counter(d.metadata.get('sub_category') for d in self.documents),
}
```

---

## 7. 静态方法

```python
@staticmethod
def get_supported_categories() -> Dict[str, Dict]:
    # 返回关键词 -> filter字典，供主程序_extract_filters_from_query用
    return {
        "眼科": {"category": "disease", "sub_category": "眼科"},
        "降压药": {"category": "drug", "sub_category": "降压药"},
        "抗生素": {"category": "drug", "sub_category": "抗生素"},
        "指南": {"category": "guideline"},
        "科普": {"category": "article"},
        # ...按实际数据目录补
    }

@staticmethod
def get_supported_difficulties() -> List[str]:
    return ["危重", "严重", "中等", "轻微"]
```
