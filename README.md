# medical-rag

眼科/视光领域的 RAG（Retrieval-Augmented Generation）检索问答系统。基于 38 份眼科学、视光学专业教材、职业资格资料，提供专业的眼科/视光知识问答。

---

## 功能特性

- **双路混合检索**：FAISS 向量检索（BAAI/bge-small-zh-v1.5）+ BM25 关键词检索，通过 RRF（Reciprocal Rank Fusion）去重融合，兼顾语义和关键词召回
- **分结构切块**：Markdown 标题层级优先切块 + 递归字符切块，父子分块反查保留大段原文
- **三阶段流式问答**：Query 路由分类（list/detail/general）→ 智能重写 → 生成，可选择流式/非流式输出
- **离线友好**：Embedding 模型走 langchain-huggingface 本地缓存，不依赖 HF_HUB_OFFLINE 环境变量
- **元数据过滤**：支持按资料分类（眼科学/视光/健康科普、难度等级等字段筛选检索范围

---

## 技术栈

| 模块 | 选型 |
|---|---|
| 语言 | Python 3.12.7 |
| RAG 框架 | LangChain 0.3.26（LCEL 语法） |
| 向量库 | FAISS-CPU |
| Embedding 模型 | BAAI/bge-small-zh-v1.5（512 维，~100MB） |
| LLM | glm-5.3-flash（AIHUBMIX OpenAI 兼容接口） |
| BM25 | rank_bm25 0.2.2 |
| 混合重排 | RRF（k=60，chunk_id 去重+MD5 兜底 |
| Temperature | 0.3（硬锁，不随用户输入变化 |

---

## 项目结构

```
medical-rag/
├── code/                          # 代码根目录
│   ├── main.py                 # 全流程入口：建库/加载索引/交互问答
│   ├── config.py             # 路径/模型/检索/生成配置
│   ├── requirements.txt        # 依赖清单
│   ├── .env.example          # API Key 配置模板
│   └── rag_modules/          # 4 大核心模块
│       ├── data_preparation.py     # 文档加载/元数据增强/结构化切块（38 docs → 12136 chunks
│       ├── index_construction.py  # Embedding 实例化 / FAISS 建库 / 索引落盘
│       ├── retrieval_optimization.py  # 双路检索 / RRF 融合 / metadata 过滤
│       └── generation_integration.py  # 路由分类/查询重写/3类生成/流式输出
├── data/                          # 38 份 MD 原始资料（已分类）
│   ├── diseases/
│   │   ├── ophthalmology/      # 眼科学教材 13 本
│   │   └── optometry/         # 视光学教材 11 本
│   └── health_articles/       # 职业资格/科普 14 本
├── vector_index/                # 本地向量索引（自动生成，不上传 git）
└── docs/                        # 开发流程文档
```

---

## 快速开始

### 1. 环境准备

```powershell
# 1.1 创建并激活 conda 环境（必须 Python 3.12.7
conda create -n medical-rag python=3.12.7 -y
conda activate medical-rag

# 1.2 安装依赖
cd code
pip install -r requirements.txt
```

### 2. 配置 API Key

```powershell
# 在 code/ 目录下
copy .env.example .env
```

然后编辑 `.env`，填入你的 AIHUBMIX Key：

```env
AIHUBMIX_API_KEY=sk-xxxxxxxxxxxxxxxx
```

### 3. 启动

```powershell
cd code
python main.py
```

首次启动后流程：

1. 检测到 `vector_index/` 不存在 → 自动建库（加载 38 份 MD → 切块 12136 条 → FAISS 建索引 → 落盘 `vector_index/`（约 1~3 分钟，索引 ~46MB
2. 检测到索引已存在 → 直接加载索引（0.5s 内）
3. 进入交互模式：

```
您的问题: 青光眼有什么症状
是否使用流式输出? (y/n, 默认y): y
```

退出交互输入 `q` / `quit` / `exit` 即可。

---

## 3 类路由说明

启动后会自动输出路由结果：

| 路由类型 | 触发场景 | 对应生成模式 |
|---|---|---|
| `list` | 问清单/列表/汇总（如「眼科常见疾病有哪些」 | 条目清单式回答 |
| `detail` | 问症状/原理/操作流程（如「配眼镜完整流程」） | 分步骤详细说明 |
| `general` | 其他一般性科普/生活咨询 | 自由格式详细回答 |

---

## 常见问题

| 问题 | 解决 |
|---|---|
| ModuleNotFound: `langchain_openai` 或 `rank_bm25` | 重新在 `code/` 目录执行 `pip install -r requirements.txt` 确保完整安装 |
| `.env` 中 Key 找不到 API Key | 检查 `code/.env` 是否存在且 `AIHUBMIX_API_KEY` 是否正确填写 |
| 索引建好后如何重新建库 | 直接删除 `vector_index/` 文件夹，重新 `python main.py` |

---

## 检索参数（硬编码常量

```
向量检索 k = 15
BM25 检索 k = 15
RRF 融合 k = 60
过滤扩展乘数 FILTER_MUL = 3
最终切片 top_k = 3
上下文构建 max_chars = 2000
```
