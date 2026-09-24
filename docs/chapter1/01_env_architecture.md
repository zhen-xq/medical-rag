# 01 环境与架构

## 1. 环境搭建

```bash
conda create -n medical-rag python=3.12.7
conda activate medical-rag
cd f:/StudyMaterials/medical-rag/code/medical_rag
pip install -r requirements.txt
```

## 2. API配置

复制 `.env.example` → `.env`，填入：
```
AIHUBMIX_API_KEY=sk-xxx
```

## 3. 架构总览（5步流水线）

```
用户提问
  → ①安全预检（命中危重词→直接输出120提示）
  → ②查询路由（5分类：disease_list / disease_detail / drug_info / treatment_info / general）
  → ③查询重写（list类跳过，其他用LLM优化检索词）
  → ④检索：向量(top15) + BM25(top15) → RRF融合(k=60) → 截top_k
     → 子块 → get_parent_documents（按命中次数去重排序）→ 父文档列表
  → ⑤生成：前置免责声明 + 按路由类型选5种Prompt模板之一 → LLM输出
```

## 4. 项目结构清单

```
medical-rag/
├── docs/chapter1/
│   ├── 01_env_architecture.md
│   ├── 02_data_preparation.md
│   ├── 03_index_retrieval.md
│   └── 04_generation_sys.md
├── code/medical_rag/
│   ├── config.py                 MedicalRAGConfig（路径/模型/top_k/temperature/max_tokens）
│   ├── main.py                   MedicalRAGSystem主类（initialize_system/build_knowledge_base/ask_question/run_interactive）
│   ├── requirements.txt
│   ├── .env.example
│   └── rag_modules/
│       ├── __init__.py           导出4个Module类
│       ├── data_preparation.py   DataPreparationModule（加载/增强/分块/父子映射/去重）
│       ├── index_construction.py IndexConstructionModule（BGE嵌入/FAISS构建/保存/加载）
│       ├── retrieval_optimization.py RetrievalOptimizationModule（向量+BM25/RRF重排/过滤检索）
│       └── generation_integration.py GenerationIntegrationModule（安全/路由/重写/5模式生成）
├── data/                         放Markdown医疗文档（子目录分类：diseases/ drugs/ guidelines/ articles）
└── vector_index/                 FAISS缓存（首次构建后自动生成）
```

## 5. 父子分块策略

```
父文档 = 完整疾病/药品/指南Markdown（doc_type=parent，parent_id=uuid）
  │
  ├── 子块0：# 标题 + ## 第一大节（doc_type=child，继承父元数据 + chunk_id + parent_id + chunk_index）
  ├── 子块1：## 第二大节
  ├── 子块2：## 第三大节 ...
  │
  └── 存储在 self.parent_child_map[chunk_id] = parent_id

检索阶段：用子块做精确匹配（小）
生成阶段：子块→查parent_id→拿完整父文档（大）
去重：同一父文档多个子块命中→只保留1份父文档，按命中次数降序排序
```

## 6. 元数据字段

| 字段 | 取值 |
|-----|------|
| `doc_type` | parent / child |
| `parent_id` / `chunk_id` | uuid |
| `category` | disease / drug / guideline / article |
| `sub_category` | 心血管/呼吸/抗生素/降压药/... |
| `source_level` | A(指南) / B(教材/说明书) / C(科普) |
| `doc_title` | 文档标题（疾病名/药品名） |
| `severity` | critical / moderate / mild |
| `source` | 文件路径 |
