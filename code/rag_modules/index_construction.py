"""
索引构建模块 - 负责向量化和FAISS索引构建
"""

import sys
import time
import warnings
from typing import List, Optional
from pathlib import Path
from langchain_core.documents import Document

# 忽略 LangChain 0.3.x 对老 embedding 兼容层的 Deprecation/Future 警告，
# 保持输出干净，不被 WARNING 刷屏。
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*deprecated.*", category=FutureWarning)


class IndexConstructionModule:
    """索引构建模块 - 负责文本向量化和FAISS索引构建"""

    # =============================================================
    # 「模型加载」的一组常量化配置，跟 __init__ 参数一一对应（不写死在方法里，方便将来在 config.py 统一改）
    # 将来如果换模型（比如换成 bge-large-zh-v1.5），只需要改 __init__ 里传的 model_name + 可选下面 2 个 DEVICE/NORMALIZE
    # =============================================================
    DEFAULT_DEVICE = "cpu"
    DEFAULT_NORMALIZE_EMBEDDINGS = True
    DEFAULT_BATCH_SIZE = 32

    def __init__(self, model_name: str = "BAAI/bge-small-zh-v1.5",
                 index_save_path: str | None = None):
        # 模型名（HuggingFace Hub ID 或本地文件夹路径，比如你手动把 bge-small-zh-v1.5 下到本地了）
        self.model_name = model_name

        # =============================================================
        # 【关键路径修正】index_save_path 默认值不写死相对路径 "./vector_index"（会依赖 cwd 跑到奇怪的地方）
        #   而是根据 __file__ 解析：当前文件位于 medical-rag/code/rag_modules/index_construction.py
        #   往上 2 层就是项目根目录（medical-rag/）—— vector_index/ 永远放在项目根下，在哪跑都一样
        # =============================================================
        if index_save_path is None:
            current_file = Path(__file__).resolve()             # .../medical-rag/code/rag_modules/index_construction.py
            project_root = current_file.parents[2]              # parents[0]=rag_modules, parents[1]=code, parents[2]=medical-rag 项目根
            index_save_path = str(project_root / "vector_index")
        # FAISS 索引落盘/加载的文件夹路径（统一转 Path 对象，后面用 mkdir/save 都方便）
        self.index_save_path = Path(index_save_path)

        # 两个核心实例变量，在 setup_embeddings() / build_vector_index() 之后分别变成：LangChain Embeddings 对象 / LangChain FAISS Vectorstore 对象
        self.embeddings = None
        self.vectorstore = None

    def setup_embeddings(self):
        """
        初始化嵌入模型（幂等：已加载就直接返回，不重复 new）。
        核心职责：把 sentence-transformers 里的 BAAI/bge-small-zh-v1.5 包成 LangChain 能认的 `embeddings` 接口对象，
        这样后面 FAISS.from_texts / as_retriever / similarity_search 都直接传它就行，不用每次手动写 encode 代码。
        """
        # =============================================================
        # ① 幂等保护：已经加载过了，直接 return（节省几秒 + 内存）
        #    （场景：用户反复调 build_index，或者重启程序后 load_index 又调 setup_embeddings——重复加载纯浪费）
        # =============================================================
        if self.embeddings is not None:
            print(f"[IndexConstruction.setup_embeddings] embeddings 已存在（模型={self.model_name}），跳过重复加载")
            return

        # =============================================================
        # ② 尝试 import HuggingFaceEmbeddings（langchain-huggingface 新包，和 C8 一致）
        #    （它内部同样基于 sentence-transformers，第一次下载缓存后不做远程 HEAD 校验，
        #     彻底解决每次新进程卡 modules.json 超时重试的问题）
        # =============================================================
        t0 = time.time()
        try:
            from langchain_huggingface import HuggingFaceEmbeddings
        except ImportError as e:
            # ImportError 基本就是「没装 langchain-huggingface / sentence-transformers」
            print(
                "[ERROR][IndexConstruction.setup_embeddings] 缺少依赖库 langchain-huggingface 或 sentence-transformers。\n"
                "  请先在当前 conda env 里执行：\n"
                "    pip install langchain-huggingface sentence-transformers faiss-cpu\n"
                f"  原始 ImportError 信息：{e}"
            )
            raise   # 不吞，还是把原始异常往上抛（main.py 可以决定怎么处理）

        # =============================================================
        # ③ 组装 HuggingFaceEmbeddings 的 3 组构造参数
        #    注意：langchain-huggingface 里的 encode_kwargs 不接受 batch_size，只有
        #          sentence-transformers 的 encode() 本身才支持 batch_size；
        #          HuggingFaceEmbeddings 内部会按批次 encode，我们只传 normalize 即可。
        # =============================================================
        model_kwargs = {
            # "device": "cpu"：强制走 CPU 计算（不依赖 torch cuda，兼容性最好；未来换 GPU 只要改成 "cuda"）
            "device": self.DEFAULT_DEVICE,
        }
        encode_kwargs = {
            # normalize_embeddings=True：encode 出来的向量自动做 L2 归一化（范数 == 1）
            # 好处：余弦相似度 = 向量点积 = L2 距离的单调函数，FAISS 的 IndexFlatL2 直接能用
            "normalize_embeddings": self.DEFAULT_NORMALIZE_EMBEDDINGS,
        }

        # =============================================================
        # ④ 真正实例化 HuggingFaceEmbeddings（第一次调这句话时 sentence-transformers 会：
        #    - 从 HuggingFace Hub 拉取 BAAI/bge-small-zh-v1.5（~100MB）到本地 cache；
        #    - 用 torch 加载权重到 CPU；
        #    - 这个过程第一次会慢（几十秒到几分钟，取决于网速和 CPU）；
        #    - 后续有本地缓存后，langchain-huggingface 不会再做远程 HEAD 校验，秒启动。
        # =============================================================
        print(f"[IndexConstruction.setup_embeddings] 开始加载 Embedding 模型：{self.model_name}")
        print(f"    device={model_kwargs['device']}, normalize={encode_kwargs['normalize_embeddings']}, batch_size={self.DEFAULT_BATCH_SIZE}")
        try:
            self.embeddings = HuggingFaceEmbeddings(
                model_name=self.model_name,            # HuggingFace ID
                model_kwargs=model_kwargs,              # device + 其它 torch 加载参数
                encode_kwargs=encode_kwargs,            # normalize
            )
        except (OSError, ConnectionError) as e:
            # OSError / ConnectionError 基本 99% 是「下模型时没网 / 连不上 HuggingFace（被墙）/ 本地模型路径不存在」
            print(
                "\n[ERROR][IndexConstruction.setup_embeddings] 模型加载失败：无法从 HuggingFace Hub 获取模型权重或读取本地模型。\n"
                "  可能的原因和解决方案：\n"
                "    1) 网络不通（国内访问 HuggingFace 慢/失败）→ 挂代理，或手动从镜像站下载模型放到本地目录，然后把\n"
                "       config.py 里的 embedding 改成「绝对路径模型文件夹」（如 D:/models/bge-small-zh-v1.5）\n"
                "    2) 手动给的 model_name 路径写错了 → 检查文件是否存在\n"
                f"  原始错误：{type(e).__name__}: {e}\n"
            )
            raise
        except Exception as e:
            # 兜底：其他任何未知错误（比如 torch 版本不兼容、model.safetensors 坏了等）
            print(
                f"\n[ERROR][IndexConstruction.setup_embeddings] 模型加载失败，未知错误：\n"
                f"  {type(e).__name__}: {e}\n"
            )
            raise

        # =============================================================
        # ⑤ 成功，打印耗时（给第一次下载心里有数），结束
        # =============================================================
        dt = time.time() - t0
        print(f"[IndexConstruction.setup_embeddings] ✅ 模型加载成功！耗时 {dt:.1f}s（模型={self.model_name}）")

    def build_vector_index(self, chunks):
        """
        构建 FAISS 向量索引（最耗时的一步：对所有 chunks 做 embedding 写入索引）。

        核心流程：
          1) 幂等调用 setup_embeddings() —— 确保 self.embeddings 已就绪
          2) 防御：入参 chunks 为空 → 直接 return + 告警
          3) 从 chunks 里抽出 texts（所有 chunk.page_content）和 metadatas（所有 chunk.metadata）
          4) 调 LangChain FAISS.from_texts(texts, embedding, metadatas) 构建向量库
          5) 结果赋值给 self.vectorstore，并打印完成耗时 / index 大小

        参数:
            chunks: List[Document] —— DataPreparationModule.chunk_documents() 产出的子块列表
        返回:
            FAISS Vectorstore 对象（即 self.vectorstore，方便链式调用）
        """
        # =============================================================
        # ① 先把 embedding 模型加载好（幂等，已加载就秒 return）
        # =============================================================
        self.setup_embeddings()

        # =============================================================
        # ② 入参合法性检查：chunks 不能是空的，否则 FAISS.from_texts 会炸
        # =============================================================
        if not chunks:
            print("[WARN][IndexConstruction.build_vector_index] 传入的 chunks 为空，跳过构建索引")
            return self.vectorstore   # 直接 return（vectorstore 还是 None）

        t0 = time.time()
        n = len(chunks)
        # =============================================================
        # ③ 抽文本和元数据
        #    - texts: List[str]，每个 chunk 的 page_content
        #    - metadatas: List[Dict]，每个 chunk 的 metadata（将来 FAISS.similarity_search 返回的 doc 里会带着）
        # =============================================================
        texts     = [c.page_content for c in chunks]
        metadatas = [c.metadata     for c in chunks]

        # 防御：langchain FAISS.from_texts 要求 texts 和 metadatas 等长，否则报错（提前打中文提示）
        if len(texts) != len(metadatas):
            raise ValueError(
                f"[ERROR][IndexConstruction.build_vector_index] texts 长度={len(texts)} ≠ metadatas 长度={len(metadatas)}！"
                " 请检查 chunks 是否有某个 Document 缺失了 page_content 或 metadata。"
            )

        print()
        print("=" * 70)
        print(f"[IndexConstruction.build_vector_index] 开始构建 FAISS 向量索引：共 {n} 个 chunks")
        print(f"    平均每 chunk 长度：{round(sum(len(t) for t in texts) / n, 0)} 字符")
        print(f"    模型：{self.model_name}（batch_size={self.DEFAULT_BATCH_SIZE}，device={self.DEFAULT_DEVICE}）")
        print("=" * 70)

        # =============================================================
        # ④ 尝试 import FAISS（缺库给清晰中文提示 + 给可执行 pip 命令）
        # =============================================================
        try:
            from langchain_community.vectorstores import FAISS
        except ImportError as e:
            print(
                "[ERROR][IndexConstruction.build_vector_index] 缺少依赖库 faiss-cpu。\n"
                "  请先执行：\n"
                "    pip install faiss-cpu langchain-community\n"
                f"  原始 ImportError 信息：{e}"
            )
            raise

        # =============================================================
        # ⑤ 调用 FAISS.from_texts 建索引（内部会调用 self.embeddings.embed_documents 批量 encode）
        #    这个函数内部做了：
        #      - 把 texts 按 batch_size 分批 encode 成 (N, 512) 矩阵
        #      - 建 FAISS IndexFlatL2（L2 距离 = 余弦相似度单调变换，因为 normalize=True）
        #      - 把 metadatas 和文本对应存到内部 docstore
        # =============================================================
        try:
            self.vectorstore = FAISS.from_texts(
                texts=texts,
                embedding=self.embeddings,
                metadatas=metadatas,
            )
        except Exception as e:
            print(
                f"\n[ERROR][IndexConstruction.build_vector_index] FAISS.from_texts 构建失败：\n"
                f"  {type(e).__name__}: {e}\n"
            )
            raise

        # =============================================================
        # ⑥ 成功：打印耗时 / 估算索引大小（docstore 的 key 数 == chunks 数就对了）
        # =============================================================
        dt = time.time() - t0
        print(f"\n[IndexConstruction.build_vector_index] ✅ 构建完成！耗时 {dt:.1f}s，索引内含 {len(self.vectorstore.docstore._dict)} 条文档（应 == {n} chunks）")

        return self.vectorstore

    def save_index(self):
        """
        保存内存里的 vectorstore 到 self.index_save_path 文件夹（默认项目根下 vector_index/）。
        落盘两个文件：
          - index.faiss    ← FAISS 向量结构（二进制）
          - index.pkl      ← docstore + metadata（pickle）
        """
        # =============================================================
        # ① 防御：还没 build 就 save，直接报错（别静默吞，给中文提示）
        # =============================================================
        if self.vectorstore is None:
            raise RuntimeError(
                "[ERROR][IndexConstruction.save_index] self.vectorstore 还没构建！"
                " 请先调用 build_vector_index(chunks) 或 load_index()。"
            )

        t0 = time.time()
        save_dir = self.index_save_path
        # =============================================================
        # ② parents=True：连父目录一起建（vector_index/ 不存在也没关系）；exist_ok=True：存在也不报错
        # =============================================================
        save_dir.mkdir(parents=True, exist_ok=True)

        print(f"[IndexConstruction.save_index] 开始保存 FAISS 索引到：{save_dir}")
        try:
            self.vectorstore.save_local(str(save_dir))
        except Exception as e:
            print(
                f"\n[ERROR][IndexConstruction.save_index] 保存失败（常见原因：磁盘满、目录不可写、磁盘意外断开）：\n"
                f"  {type(e).__name__}: {e}\n"
            )
            raise

        # =============================================================
        # ③ 统计落盘大小（给用户心里有数，一般 12136 条 chunks 大概 40~60MB）
        # =============================================================
        total_bytes = 0
        for p in save_dir.iterdir():
            if p.is_file():
                total_bytes += p.stat().st_size
        total_mb = total_bytes / (1024 * 1024)
        dt = time.time() - t0
        print(f"[IndexConstruction.save_index] ✅ 保存成功！耗时 {dt:.1f}s，总大小 {total_mb:.2f} MB，文件列表：")
        for p in sorted(save_dir.iterdir()):
            if p.is_file():
                mb = p.stat().st_size / (1024 * 1024)
                print(f"      - {p.name}: {mb:.2f} MB")

    def load_index(self):
        """
        从 self.index_save_path 文件夹里把之前 save_local 过的 FAISS 索引读回到内存里。
        幂等：如果 self.vectorstore 已经加载过了，直接 return 不重复加载。

        【巨坑注意】FAISS 新版（langchain>=0.3 里包的版本）出于安全考虑，默认禁止反序列化 pickle（index.pkl），
        会抛 ValueError("The de-serialization relies on loading a pickle file... allow_dangerous_deserialization=True")。
        因为这个索引是我们自己本地生成的（不是第三方给的），绝对安全，所以必须传 allow_dangerous_deserialization=True！
        不写这句 100% 崩。
        """
        # =============================================================
        # ① 幂等：vectorstore 已经有了就直接 return
        # =============================================================
        if self.vectorstore is not None:
            print(f"[IndexConstruction.load_index] vectorstore 已存在（{len(self.vectorstore.docstore._dict)} 条），跳过重复加载")
            return self.vectorstore

        # =============================================================
        # ② 先加载 embedding 模型（FAISS.load_local 需要 embeddings 对象来做后续的查询 encode）
        # =============================================================
        self.setup_embeddings()

        load_dir = self.index_save_path
        print(f"[IndexConstruction.load_index] 开始从 {load_dir} 加载 FAISS 索引")

        # =============================================================
        # ③ 防御：目录不存在 / 里面没有 index.faiss 就给清晰中文提示
        # =============================================================
        if not load_dir.exists() or not load_dir.is_dir():
            raise FileNotFoundError(
                f"[ERROR][IndexConstruction.load_index] 索引目录不存在：{load_dir}\n"
                f"  请先运行 build_vector_index(chunks) + save_index() 生成索引。"
            )
        faiss_file = load_dir / "index.faiss"
        pkl_file   = load_dir / "index.pkl"
        if not faiss_file.exists() or not pkl_file.exists():
            raise FileNotFoundError(
                f"[ERROR][IndexConstruction.load_index] 索引目录里缺少必需文件：{load_dir}\n"
                f"  需要同时存在 index.faiss（当前：{faiss_file.exists()}）和 index.pkl（当前：{pkl_file.exists()}）\n"
                f"  请先删除该目录后重新 build_vector_index + save_index。"
            )

        t0 = time.time()
        # import FAISS
        try:
            from langchain_community.vectorstores import FAISS
        except ImportError as e:
            print(
                "[ERROR][IndexConstruction.load_index] 缺少依赖库 faiss-cpu / langchain-community。\n"
                "  请执行：pip install faiss-cpu langchain-community\n"
                f"  ImportError: {e}"
            )
            raise

        # =============================================================
        # ④ 加载（allow_dangerous_deserialization=True —— 新版 FAISS 强制要这句，否则崩）
        # =============================================================
        try:
            self.vectorstore = FAISS.load_local(
                folder_path=str(load_dir),
                embeddings=self.embeddings,
                allow_dangerous_deserialization=True,   # ⚠️ 新版 FAISS 必传！否则 ValueError 直接崩
            )
        except Exception as e:
            print(
                f"\n[ERROR][IndexConstruction.load_index] FAISS.load_local 加载失败：\n"
                f"  {type(e).__name__}: {e}\n"
                f"  常见修复：如果上面提示 allow_dangerous_deserialization，请确保代码里传了这个参数=True\n"
            )
            raise

        # =============================================================
        # ⑤ 成功：打印加载条数、大小、耗时
        # =============================================================
        n_docs = len(self.vectorstore.docstore._dict)
        dt = time.time() - t0
        total_mb = sum(p.stat().st_size for p in load_dir.iterdir() if p.is_file()) / (1024 * 1024)
        print(f"[IndexConstruction.load_index] ✅ 加载成功！耗时 {dt:.1f}s，共 {n_docs} 条文档，磁盘总大小 {total_mb:.2f} MB")

        return self.vectorstore
