# 04 生成集成 + 系统整合

## A. GenerationIntegrationModule
**文件**: `code/medical_rag/rag_modules/generation_integration.py`

### 类成员
```python
self.model_name = "glm-5.3-flash"
self.temperature = 0.3
self.max_tokens = 2048
self.llm: ChatOpenAI | None
```

---

### 1. setup_llm()

```python
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os
load_dotenv()

self.llm = ChatOpenAI(
    api_key=os.getenv("AIHUBMIX_API_KEY"),
    base_url="https://aihubmix.com/v1",
    model=self.model_name,
    temperature=self.temperature,
    max_tokens=self.max_tokens
)
```
`__init__` 中立刻调用。

---

### 2. query_router(query: str) -> str （5分类）

**合法返回值**: `disease_list` / `disease_detail` / `drug_info` / `treatment_info` / `general`

**LCEL链**:
```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

ROUTER_PROMPT = ChatPromptTemplate.from_template("""
把用户问题分类为5种之一，只输出类型名，不要其他文字。

1. disease_list: 要列表/类别大全
   例：眼科常见疾病？视光类书籍有哪些？

2. disease_detail: 了解某具体主题详情
   例：青光眼有什么症状？近视有哪些矫正方式？

3. drug_info: 药品信息/用药问题
   例：左氧氟沙星怎么吃？阿托品的作用？

4. treatment_info: 流程/指导/步骤
   例：体检发现近视怎么办？配眼镜要做哪些检查？

5. general: 其他科普/咨询
   例：怎么保护眼睛？儿童视力发育规律？

用户问题: {query}
分类结果:""")

chain = ROUTER_PROMPT | self.llm | StrOutputParser()
result = chain.invoke({"query": query}).strip()
if result not in ["disease_list","disease_detail","drug_info","treatment_info","general"]:
    result = "general"
return result
```

---

### 3. query_rewrite(query: str, route_type: str) -> str

**规则**:
- `disease_list` → 直接 `return query`（不重写）
- 其他4类 → 用LLM优化

```python
REWRITE_PROMPT = ChatPromptTemplate.from_template("""
把用户查询改写为精准检索词。
- 俗称转标准名
- 保留原意，补充相关关键词
- 只输出改写结果，不解释

原问题类型: {route_type}
原问题: {query}
改写后查询:""")

chain = REWRITE_PROMPT | self.llm | StrOutputParser()
return chain.invoke({"route_type": route_type, "query": query}).strip()
```

---

### 4. 上下文拼装工具（公用）

```python
def _build_context(self, context_docs: List[Document]) -> str:
    level_map = {"A": "A级资料", "B": "B级资料", "C": "C级科普"}
    parts = []
    for i, doc in enumerate(context_docs, 1):
        lvl = doc.metadata.get("source_level", "B")
        title = doc.metadata.get("doc_title", f"资料{i}")
        parts.append(f"[{i}] {title}（{level_map.get(lvl, 'B级')}）：\n{doc.page_content}")
    return "\n\n".join(parts)
```

---

### 5. 5种生成模式（每个含普通版 + _stream版）

#### 5.1 generate_list_answer(query, context_docs) / _stream
**适用**: disease_list
**不调LLM，纯模板拼接**：
```
1. 从 context_docs 提取 (sub_category, doc_title)，去重
2. 按 sub_category 分组
3. 输出格式：
   根据知识库，整理相关条目：

   【眼科】
   1. 白内障
   2. 青光眼
   3. 近视

   【视光学】
   4. 散光
   5. 弱视

   如需详情请输入具体主题名。
```
Stream版：逐行yield分类列表。

---

#### 5.2 generate_step_by_step_answer(query, context_docs) / _stream
**适用**: disease_detail
```python
DETAIL_PROMPT = ChatPromptTemplate.from_template("""
严格基于【知识库资料】回答，资料中没有的就写"暂无详细资料"，不要编造。
按以下6个小标题输出：
📖 概述
🩺 典型表现
🔬 诊断与检查
💊 干预/矫正方案
📈 预后与转归
🛡️ 预防与日常养护

用户问题: {query}
【知识库资料】
{context}
【回答】""")

ctx = self._build_context(context_docs)
chain = DETAIL_PROMPT | self.llm | StrOutputParser()
return chain.invoke({"query": query, "context": ctx})
# Stream版本：for chunk in chain.stream(...): yield chunk
```

---

#### 5.3 generate_drug_answer(query, context_docs) / _stream
**适用**: drug_info
```python
DRUG_PROMPT = ChatPromptTemplate.from_template("""
严格基于【药品资料】，仿药品说明书8项输出（无则写"详见说明书/资料"）：
🏷️ 药品名称
🎯 适应症
💊 用法用量
⚠️ 不良反应
🚫 禁忌
🧑‍⚕️ 注意事项
🤰 特殊人群（孕妇/儿童/老人）
🔀 药物相互作用

用户问题: {query}
【药品资料】
{context}
【回答】""")
```

---

#### 5.4 generate_treatment_answer(query, context_docs) / _stream
**适用**: treatment_info
```python
TREAT_PROMPT = ChatPromptTemplate.from_template("""
按4步流程回答：
🏥 第一步：就医/咨询建议（挂什么科 / 何时需要专业检查）
🔬 第二步：可能的检查项目
💊 第三步：常规干预/矫正/治疗方案
📅 第四步：复查与随访

用户问题: {query}
【知识库资料】
{context}
【回答】""")
```

---

#### 5.5 generate_basic_answer(query, context_docs) / _stream
**适用**: general
```python
BASIC_PROMPT = ChatPromptTemplate.from_template("""
用通俗易懂的话回答，分3-5个要点，必要时对专业术语加括号解释。

用户问题: {query}
【知识库资料】
{context}
【回答】""")
```

---

## B. main.py MedicalRAGSystem

### 类成员
```python
self.config = config or DEFAULT_CONFIG
self.data_module: DataPreparationModule | None
self.index_module: IndexConstructionModule | None
self.retrieval_module: RetrievalOptimizationModule | None
self.generation_module: GenerationIntegrationModule | None
```
`__init__` 只做：检查 data_path 存在 + 检查 `AIHUBMIX_API_KEY` 环境变量。

---

### 1. initialize_system()

```python
self.data_module = DataPreparationModule(self.config.data_path)
self.index_module = IndexConstructionModule(
    model_name=self.config.embedding_model,
    index_save_path=self.config.index_save_path
)
self.generation_module = GenerationIntegrationModule(
    model_name=self.config.llm_model,
    temperature=self.config.temperature,
    max_tokens=self.config.max_tokens
)
```

---

### 2. build_knowledge_base()

```
1. vectorstore = self.index_module.load_index()

2. if vectorstore is not None:   # 命中缓存
     self.data_module.load_documents()
     chunks = self.data_module.chunk_documents()

3. else:                         # 首次，全新构建
     self.data_module.load_documents()
     chunks = self.data_module.chunk_documents()
     vectorstore = self.index_module.build_vector_index(chunks)
     self.index_module.save_index()

4. self.retrieval_module = RetrievalOptimizationModule(vectorstore, chunks)

5. 打印 get_statistics() 统计
```

---

### 3. _extract_filters_from_query(query) -> Dict

```python
cats = DataPreparationModule.get_supported_categories()
for kw, rule in cats.items():
    if kw in query:
        return rule
return {}
```

---

### 4. ask_question(question: str, stream=False)

```
# ① 查询路由
route = self.generation_module.query_router(question)

# ② 查询重写
if route == "disease_list":
    rewritten = question
else:
    rewritten = self.generation_module.query_rewrite(question, route)

# ③ 检索
filters = self._extract_filters_from_query(question)
if filters:
    chunks = self.retrieval_module.metadata_filtered_search(rewritten, filters, self.config.top_k)
else:
    chunks = self.retrieval_module.hybrid_search(rewritten, self.config.top_k)

# 空结果兜底
if not chunks:
    return "😔 未找到相关信息，建议尝试更精准的关键词。"

# ④ 子块 → 父文档（去重）
docs = self.data_module.get_parent_documents(chunks)

# ⑤ 按路由分派生成模式
gen = self.generation_module
route_map = {
    "disease_list":   (gen.generate_list_answer,            gen.generate_list_answer_stream),
    "disease_detail": (gen.generate_step_by_step_answer,    gen.generate_step_by_step_answer_stream),
    "drug_info":      (gen.generate_drug_answer,            gen.generate_drug_answer_stream),
    "treatment_info": (gen.generate_treatment_answer,       gen.generate_treatment_answer_stream),
    "general":        (gen.generate_basic_answer,           gen.generate_basic_answer_stream),
}
normal_fn, stream_fn = route_map.get(route, route_map["general"])
if stream:
    return stream_fn(question, docs)
else:
    return normal_fn(question, docs)
```

---

### 5. run_interactive()

```
打印欢迎横幅
initialize_system()
build_knowledge_base()

循环:
    user_input = input("📚 请输入问题（退出结束）：")
    if user_input in ["退出", "quit", "exit", ""]: break
    choice = input("流式输出？(y/n，默认y)：")
    use_stream = choice != 'n'
    print(40*"=" + "回答" + 40*"=")
    if use_stream:
        for chunk in ask_question(user_input, stream=True):
            print(chunk, end="", flush=True)
        print()
    else:
        print(ask_question(user_input, stream=False))
```

---

### 6. main()

```python
def main():
    try:
        MedicalRAGSystem().run_interactive()
    except Exception as e:
        print(f"❌ 错误: {e}")
        print("检查项：1.data路径 2.API Key 3.网络")

if __name__ == "__main__":
    main()
```
