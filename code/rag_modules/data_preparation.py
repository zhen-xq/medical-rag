"""
数据准备模块 - 负责医疗文档加载、清洗和预处理
"""

import uuid
from typing import List, Dict
from pathlib import Path
from langchain_core.documents import Document


class DataPreparationModule:
    """数据准备模块 - 负责医疗文档加载、元数据增强和分块处理"""

    def __init__(self, data_path: str):
        # ===== 把传入的路径字符串转成 Path 对象（方便后面 rglob / parent 等操作）
        self.data_path = Path(data_path).resolve()

        # ===== 父文档容器：一份 md 就是一个 Document（后续"大块生成"就靠它，不切分）
        self.documents: List[Document] = []

        # ===== 子文档容器：按 Markdown 标题 / 字符长度切出来的"小块"
        # 真正被丢进 FAISS 做向量索引的就是这些子块（小块检索 → 大块生成）
        self.chunks: List[Document] = []

        # ===== 父子映射：key = 子块 chunk_id（str）, value = 父文档 parent_id（str）
        # O(1) 时间通过子块反查回父文档（hit_count 去重就靠它）
        self.parent_child_map: Dict[str, str] = {}

    def load_documents(self) -> List[Document]:
        """
        加载 data/ 目录下所有 .md 文件，一份 md → 一个 Document 对象
        填入基础 metadata（source / doc_title / category / sub_category / parent_id）
        再把所有 Document 收集到 self.documents 并返回。
        """
        # =============================================================
        # ① 先清空容器（防止重复调用导致重复积累）
        # =============================================================
        self.documents.clear()
        self.chunks.clear()
        self.parent_child_map.clear()

        # =============================================================
        # ② 安全校验：data_path 必须真实存在且是目录（用户传错路径要提前报错）
        # =============================================================
        if not self.data_path.exists() or not self.data_path.is_dir():
            raise FileNotFoundError(
                f"数据目录不存在或不是目录: {self.data_path}"
            )

        # =============================================================
        # ③ 递归扫描所有 .md 文件（rglob 会自动进子目录如 diseases/ophthalmology）
        # sorted 是为了保证每次加载顺序一致（调试方便、parent_id 映射可复现）
        # =============================================================
        md_files = sorted(self.data_path.rglob("*.md"))

        for idx, md_path in enumerate(md_files):
            # ---------------------------------------------------------
            # ③-a 读取文件全文；遇到编码异常直接忽略掉那些坏字符（errors="ignore"），避免整份挂掉
            # ---------------------------------------------------------
            try:
                raw_text = md_path.read_text(encoding="utf-8", errors="ignore")
            except Exception as e:
                # 真挂了就打个日志跳过，不要让一两个坏文件搞垮整个 38 份加载
                print(f"[load_documents] WARN: 读取失败 {md_path.name}，已跳过: {e}")
                continue

            # 全文如果是空白文件（全是换行/空格），也跳过，别占坑
            if not raw_text.strip():
                print(f"[load_documents] WARN: 文件为空 {md_path.name}，已跳过")
                continue

            # ---------------------------------------------------------
            # ③-b 从"相对 data_path 的路径"反推 category / sub_category
            #   例：self.data_path/data/diseases/ophthalmology/青光眼.md
            #       rel_parts = ["diseases", "ophthalmology", "青光眼.md"]
            #       category     = "diseases"
            #       sub_category = "ophthalmology"
            # ---------------------------------------------------------
            rel_path = md_path.relative_to(self.data_path)  # 相对路径（不含 data/ 自己）
            rel_parts = list(rel_path.parts)                 # 按目录拆成列表

            category = "unknown"
            sub_category = "unknown"

            if len(rel_parts) >= 3:
                # 典型：diseases/ophthalmology/xxx.md   → 3 段
                category     = rel_parts[0]
                sub_category = rel_parts[1]
            elif len(rel_parts) == 2:
                # 典型：health_articles/门诊笔记.md    → 2 段（没有 sub_category 层）
                category     = rel_parts[0]
                sub_category = category  # 没子分类就把 category 当 sub，避免空值
            else:
                # 真直接丢 data/ 根目录的 md 就算 unknown（后续 _enhance_metadata 还能再救）
                category = "root"
                sub_category = "root"

            # ---------------------------------------------------------
            # ③-c doc_title：文件名去掉 .md 即可（38 份文件我们在 v4 脚本里已经洗得很干净）
            # ---------------------------------------------------------
            doc_title = md_path.stem

            # ---------------------------------------------------------
            # ③-d parent_id：由相对 data_path 的路径稳定生成（同一份资料每次启动都相同）
            #   注意不能用 uuid4：保存的向量索引里也会带 parent_id；若每次随机生成，
            #   加载旧索引命中 chunk 后将无法反查本次内存中的父文档。
            # ---------------------------------------------------------
            relative_source = rel_path.as_posix()
            parent_id = uuid.uuid5(
                uuid.NAMESPACE_URL,
                f"medical-rag-parent-v1:{relative_source}",
            ).hex

            # ---------------------------------------------------------
            # ③-e 组装基础 metadata；后面 _enhance_metadata 会往里面继续加
            #   source_level / difficulty / severity 等更细的字段下一个方法填
            # ---------------------------------------------------------
            metadata = {
                "source":       str(md_path),     # 绝对路径，调试时能直接定位文件
                "doc_title":    doc_title,        # 人能看懂的书名（如"青光眼"）
                "category":     category,         # diseases / health_articles / guidelines
                "sub_category": sub_category,     # ophthalmology / optometry / ...
                "parent_id":    parent_id,        # 这份父文档的全局唯一主键
                "id_scheme":    "stable-path-chunk-v1",
                "doc_index":    idx,              # 加载序号，调试用
            }

            # ---------------------------------------------------------
            # ③-f 真正造 LangChain Document，塞到列表里
            # ---------------------------------------------------------
            doc = Document(page_content=raw_text, metadata=metadata)
            self.documents.append(doc)

        # =============================================================
        # ④ 全部扫完，顺手打一行汇总日志给用户看一眼（排错用）
        # =============================================================
        print(
            f"[DataPreparation] load_documents 完成：共加载 {len(self.documents)} / 扫描到 {len(md_files)} 份 md 文件"
        )

        # =============================================================
        # ⑤ 返回这个列表（接口契约里要求返回 List[Document]）
        # =============================================================
        return self.documents

    def _enhance_metadata(self, doc: Document):
        """
        增强医疗文档元数据：在 load_documents() 填好的 5 个基础字段上，
        再追加 3 个分类标签（source_level / doc_type / difficulty）。
        所有判断都基于「doc_title 关键词匹配」，简单直观，后续可随时修改规则。

        修改：直接对传入的 doc.metadata 进行 in-place 追加，不需要返回值。
        """
        # =============================================================
        # 0. 先拿几个常用变量出来，后面重复写太啰嗦
        # =============================================================
        title = doc.metadata.get("doc_title", "")   # 书名，如 "眼科学（第十版）"
        category = doc.metadata.get("category", "")  # diseases / health_articles / guidelines
        meta = doc.metadata                          # 直接操作这个 dict（in-place 修改）

        # =============================================================
        # ① source_level：资料权威等级 A / B / C
        #   思路：先把"顶级权威"拎出来算 A，剩下的 diseases 类教材算 B，科普/职业/笔记类全算 C
        # =============================================================
        # --- A 级顶级资料（命中任意一个就算 A）
        #   1) 眼科学（第十版）：人卫本科临床教材天花板
        #   2) Wills 眼科图谱 7 本：就是 data/diseases/ophthalmology/ 下这 7 个短名（注意有 2 本视网膜）
        a_level_titles = {
            "眼科学（第十版）", "眼科学第10版", "眼科学 第10版", "眼科学第十版",   # 第十版四种写法
            "小儿眼科", "神经眼科", "眼眶病", "葡萄膜炎", "青光眼",              # Wills 7 本（注意没前缀）
            "视网膜", "视网膜_1",                                              # 视网膜×2 两本
        }
        if title in a_level_titles:
            meta["source_level"] = "A"

        # --- B 级：正规教材/专项图书（非顶级但依然是"教科书"级）
        # 判定条件：
        #   1) 必须是 diseases 类（不是职业资格/科普/笔记）
        #   2) 没被算成 A
        #   （因为 diseases 类一共 24 本，扣掉 8 本 A 级，剩下 16 本全是正规教材，直接算 B 即可）
        elif category == "diseases":
            meta["source_level"] = "B"
        else:
            # --- C 级：health_articles / guidelines / unknown 全算 C
            # （14 本职业资格/门诊笔记/Q&A/科普全落这里）
            meta["source_level"] = "C"

        # =============================================================
        # ② doc_type：资料类型（用于给用户展示"引用了什么类型的资料"）
        #   atlas / textbook / vocational / notes / qa / article
        # =============================================================
        # atlas（图谱）— Wills 眼科图谱系列 7 本（跟上面 source_level A 级用同一张表）
        wills_atlas_titles = {
            "小儿眼科", "神经眼科", "眼眶病", "葡萄膜炎", "青光眼",
            "视网膜", "视网膜_1",
        }
        if title in wills_atlas_titles:
            meta["doc_type"] = "atlas"

        # vocational（职业资格培训教材）— 眼镜定配工/验光员 等级系列
        elif any(kw in title for kw in ["眼镜定配工", "眼镜验光员"]):
            meta["doc_type"] = "vocational"

        # notes（笔记/手札）— 门诊笔记系列
        elif "门诊笔记" in title:
            meta["doc_type"] = "notes"

        # qa（问答/Q&A 类）— 书名里直接有 Q&A/问答 或常见的 Q&A 变体
        elif any(kw in title for kw in ["Q&A", "问答", "视光Q", "Q_A", "儿童视光Q"]):
            meta["doc_type"] = "qa"

        # textbook（教材）— diseases 类剩下的全是教材（= 24 - 7 atlas = 17 本 textbook）
        elif category == "diseases":
            meta["doc_type"] = "textbook"

        # article（科普文/杂项）— health_articles 剩下没命中的
        else:
            meta["doc_type"] = "article"

        # =============================================================
        # ③ difficulty：阅读难度 beginner / intermediate / advanced
        #   简单分层：科普/初级资格 → 入门；一般教材/中级资格 → 中级；A 级资料 + 第十版 + 专项高级 → 高级
        # =============================================================
        # --- advanced 判定：命中 A 级 + 几个明确是高级专科教材的标题
        advanced_titles = set(a_level_titles) | {   # A 级（Wills 7 + 第十版）直接算 advanced，不看内容
            "临床双眼视觉学",  # 临床高级专书
            "低视力学第3版",   # 低视力学是高级专项
            "屈光手术学",      # 手术类一律 advanced
            "低视力学",        # 兜底，万一以后文件名叫"低视力学.md"
        }
        if title in advanced_titles or "硬性角膜接触镜" in title:  # RGP 类专书也是进阶内容
            meta["difficulty"] = "advanced"

        # --- beginner 判定：职业资格的初级/基础知识 + 门诊笔记 + 科普/Q&A + 儿童类
        elif (
            # 职业资格等级里的「初级 / 基础知识」关键词
            any(grade in title for grade in ["（初级）", "（基础知识）", "初级", "基础知识"])
            # 门诊笔记（写给读者/非医学生看的）
            or "门诊笔记" in title
            # 问答 / 儿童视光 / 视觉与学习 这类偏科普的
            or any(kw in title for kw in ["Q&A", "问答", "视光Q", "Q_A", "儿童视光", "视觉与学习"])
        ):
            meta["difficulty"] = "beginner"

        # --- intermediate：剩下的所有（一般教材第2/3版 + 职业资格的中级/高级/技师 都落这）
        else:
            meta["difficulty"] = "intermediate"

        # =============================================================
        # ④ 最后加两个便于调试的字段（可选，不影响主流程）
        # =============================================================
        # 标记一下这份 metadata 已经经过了 enhance 处理（=1 才说明跑过了这个方法）
        meta["enhanced"] = "1"
        # 同时把 "enhanced_at" 留个锚点，后面你加规则时可以知道哪些是新补的，哪些是旧的
        # 这里为了简单就不引 datetime 了，直接写死字符串，需要时你可自行改成时间戳
        meta["_rule_ver"] = "v1_basic_keyword"

    def chunk_documents(self) -> List[Document]:
        """
        医疗文档结构感知分块（总入口）
        本次先实现 v1 简化版：
          1) 先 _markdown_header_split（下一个方法填，没实现就 fallback 成整份一段）
          2) 对每个标题段，>1500 字符就用 RecursiveCharacterTextSplitter 切 500~1500 overlap=200
          3) 每一块生成 chunk_id、继承父文档 metadata、塞 self.chunks、写 parent_child_map
        """
        # =============================================================
        # 0. 切块全局参数（写在这里方便调，不需要单独挪到文件级常量）
        # =============================================================
        CHUNK_MAX_CHARS = 1500   # 单个块最长字符数
        CHUNK_MIN_CHARS = 500    # 二次切（RecursiveCharacter）的最短目标（实际可能略小）
        CHUNK_OVERLAP   = 200    # 二次切相邻两块的重叠字符数（防止语义切断开）

        # =============================================================
        # ① 先把旧的 chunk 数据清空（防止重复调用时累积上一次的结果）
        # =============================================================
        self.chunks.clear()
        self.parent_child_map.clear()

        # =============================================================
        # ② 防御性兜底：如果 self.documents 还是空的，先自动调 load_documents()
        # 再顺便跑一遍 _enhance_metadata，保证每个父文档的 metadata 都是增强过的
        # （这样用户哪怕忘记先调 load，直接调 chunk_documents 也能工作）
        # =============================================================
        if not self.documents:
            self.load_documents()

        for doc in self.documents:
            if doc.metadata.get("enhanced") != "1":
                self._enhance_metadata(doc)

        # =============================================================
        # ③ 逐个父文档处理
        # =============================================================
        for parent_doc in self.documents:
            parent_meta = parent_doc.metadata          # 父文档的完整 metadata（chunk 大部分字段从这里继承）
            parent_id   = parent_meta["parent_id"]     # 父文档唯一 ID，每个 chunk 都要挂一份
            doc_title   = parent_meta.get("doc_title", "(no title)")

            # ---------------------------------------------------------
            # ③-a 第一步切：按 Markdown 标题做结构感知切分
            #   _markdown_header_split 现在还没实现（下一个方法填），
            #   这里用 try/except 兜一下：
            #     - 如果它成功返回了非空列表，就用返回的标题段列表
            #     - 如果它还只是 pass / 返回空 / 抛异常，就 fallback：把整份父文档当成一个「伪标题段」
            #       包成 [{"content": 全文, "section": "__WHOLE_DOC__"}]
            # ---------------------------------------------------------
            section_items = []   # 预期格式：每个元素是 dict {"content": str, "section": str/None}
            try:
                split_result = self._markdown_header_split(parent_doc)
                if isinstance(split_result, list) and len(split_result) > 0:
                    # _markdown_header_split 如果已实现，返回值允许两种形状：
                    #   A) List[Document]（每个子文档 page_content 就是段内容）
                    #   B) List[dict[str, str]]（{"content": ..., "section": ...}）
                    for item in split_result:
                        if isinstance(item, Document):
                            section_items.append({
                                "content": item.page_content,
                                "section": item.metadata.get("section"),
                            })
                        elif isinstance(item, dict):
                            section_items.append(item)
                        else:
                            # 类型不认识，按整段跳过（保守）
                            continue
            except Exception:
                # _markdown_header_split 出错就忽略，走 fallback
                section_items = []

            # fallback：没有任何标题段就把全文当一段处理
            if not section_items:
                section_items = [{
                    "content": parent_doc.page_content,
                    "section": "__WHOLE_DOC__",
                }]

            # ---------------------------------------------------------
            # ③-b 第二步切：对每一个标题段，按长度决定「直接作为 chunk / 再用 Recursive 细切」
            #   为了不重复造轮子，这里直接用 langchain 的 RecursiveCharacterTextSplitter
            #   它的 separators 会优先按 \n\n -> \n -> 句子 -> 字 这些自然断点切，
            #   比我们自己硬截断要聪明得多
            # ---------------------------------------------------------
            try:
                from langchain_text_splitters import RecursiveCharacterTextSplitter  # 延迟 import
                r_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=CHUNK_MAX_CHARS,
                    chunk_overlap=CHUNK_OVERLAP,
                    separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""],  # 中文友好的分隔优先级
                )
            except Exception as e:
                # import 失败也要兜住（比如还没装 requirements），直接硬按字符数切（不抛异常挂）
                print(f"[chunk_documents] WARN: 无法 import RecursiveCharacterTextSplitter，将使用朴素硬切。原因：{e}")
                r_splitter = None

            # --- 判定本次有没有用到 Markdown 切（即 section_items 的第一个元素不是 "__WHOLE_DOC__" 那种 fallback 伪段）
            # 用一个标记位，后面的 split_from 字段会用到
            used_markdown_split = (
                len(section_items) >= 1
                and not (len(section_items) == 1 and section_items[0].get("section") == "__WHOLE_DOC__")
            )

            # raw_sub_pieces 从存纯字符串改成存 dict，每个元素带有：
            #   piece:      str            # 子块正文
            #   section:    str|None       # 所属 Markdown 标题链路径（从 _markdown_header_split 传过来的）
            #   split_from: str            # 本块到底是哪种来源，4 种细分值见前面文档说明
            raw_sub_pieces: list = []

            for sec in section_items:
                sec_content = sec.get("content", "") or ""
                sec_content = sec_content.strip()
                sec_section = sec.get("section")   # 可能是 None（开头无标题段）或 "__WHOLE_DOC__"（fallback 伪段）或正常标题链

                # 跳过纯空段（全是换行/空格的标题壳）
                if not sec_content:
                    continue

                # --- 短段直接留用（不经过 Recursive 二次切）
                if len(sec_content) <= CHUNK_MAX_CHARS:
                    if used_markdown_split and sec_section != "__WHOLE_DOC__":
                        tag = "markdown_header_direct"
                    else:
                        # fallback 整段但非常短（整本书特别薄），直接标 whole_doc_recursive/plain 下面按 r_splitter 再定
                        tag = "whole_doc_recursive" if r_splitter is not None else "whole_doc_plain"
                    raw_sub_pieces.append({
                        "piece":      sec_content,
                        "section":    sec_section,
                        "split_from": tag,
                    })
                    continue

                # --- 长段 → 需要二次切
                # 先定这一批子块的 split_from 前缀：用了 Markdown 切就叫 markdown_header_recursive，否则叫 whole_doc_*
                if used_markdown_split and sec_section != "__WHOLE_DOC__":
                    recursive_tag = "markdown_header_recursive"
                elif r_splitter is not None:
                    recursive_tag = "whole_doc_recursive"
                else:
                    recursive_tag = "whole_doc_plain"

                if r_splitter is not None:
                    # RecursiveCharacterTextSplitter.split_text 返回 List[str]
                    for piece in r_splitter.split_text(sec_content):
                        piece = (piece or "").strip()
                        if piece and len(piece) >= 20:   # 切出来的块太小（<20字）也不要，纯噪声
                            raw_sub_pieces.append({
                                "piece":      piece,
                                "section":    sec_section,
                                "split_from": recursive_tag,
                            })
                else:
                    # fallback：朴素硬切（按 CHUNK_MAX_CHARS - CHUNK_OVERLAP 步长切片子串）
                    step = max(1, CHUNK_MAX_CHARS - CHUNK_OVERLAP)
                    i = 0
                    while i < len(sec_content):
                        piece = sec_content[i: i + CHUNK_MAX_CHARS].strip()
                        if piece and len(piece) >= 20:
                            raw_sub_pieces.append({
                                "piece":      piece,
                                "section":    sec_section,
                                "split_from": recursive_tag,
                            })
                        i += step

            # ---------------------------------------------------------
            # ③-c 把 raw_sub_pieces 里的每一段 → 包装成正式 chunk Document
            #      并写 parent_child_map
            # ---------------------------------------------------------
            for idx_in_parent, item in enumerate(raw_sub_pieces):
                piece_text = item["piece"]
                section    = item.get("section")
                split_from = item.get("split_from", "unknown")

                # --- chunk_id：由父文档、块序号和正文稳定生成。
                #     这样 FAISS 旧索引与本次 BM25 的同一块能被 RRF 正确去重。
                chunk_id = uuid.uuid5(
                    uuid.NAMESPACE_URL,
                    f"medical-rag-chunk-v1:{parent_id}:{idx_in_parent}:{section or ''}:{piece_text}",
                ).hex

                # --- 合并 metadata：先从父文档拷贝"可继承字段"，再叠加 chunk 独有的字段
                chunk_meta = {
                    # --- 继承自父文档的分类标签
                    "category":     parent_meta.get("category", "unknown"),
                    "sub_category": parent_meta.get("sub_category", "unknown"),
                    "doc_title":    parent_meta.get("doc_title", ""),
                    "source_level": parent_meta.get("source_level", "C"),
                    "doc_type":     parent_meta.get("doc_type", "article"),
                    "difficulty":   parent_meta.get("difficulty", "intermediate"),
                    # --- 继承自父文档的溯源信息
                    "source":       parent_meta.get("source", ""),
                    "parent_id":    parent_id,            # 核心：小块 → 大块 的反查锚点
                    "id_scheme":    "stable-path-chunk-v1",
                    # --- chunk 自己的信息
                    "chunk_id":     chunk_id,             # 本块全局唯一 ID
                    "chunk_index":  idx_in_parent,        # 本块在父文档内部的序号（0起）
                    "split_from":   split_from,           # 4 种细分来源：markdown_header_direct / markdown_header_recursive / whole_doc_recursive / whole_doc_plain
                }
                # --- 把"标题链路径 section"也写进 metadata（方便调试/检索；None 就不写，省得占地方）
                if section is not None and section != "__WHOLE_DOC__":
                    chunk_meta["section"] = section

                # --- 造 LangChain Document，追加到总 chunk 池
                chunk_doc = Document(page_content=piece_text, metadata=chunk_meta)
                self.chunks.append(chunk_doc)

                # --- 写父子映射（小块id → 大块id）
                self.parent_child_map[chunk_id] = parent_id

            # 父文档结束：可以在这里加一句 per-doc 日志（数据量大再开，现在先静默）
            # print(f"  [chunk] {doc_title}: {len(raw_sub_pieces)} 子块")

        # =============================================================
        # ④ 结束汇总日志
        # =============================================================
        print(
            f"[DataPreparation] chunk_documents 完成："
            f"{len(self.documents)} 份父文档 → {len(self.chunks)} 个子块，"
            f"parent_child_map 条目数 = {len(self.parent_child_map)}"
        )

        # =============================================================
        # ⑤ 返回 chunks（接口契约要求）
        # =============================================================
        return self.chunks

    def _markdown_header_split(self, parent_doc: Document):
        """使用Markdown标题分割器进行结构化分割。
        入参：单个父文档（一份 md 全文）
        返回：
            成功 → List[dict]，每个 dict 形如：
                {"content": str（段正文，含标题行）,
                 "section": str 或 None（标题链路径，方便调试/检索增强）}
            失败 → None（外层 chunk_documents 会走「整份一段」的 fallback）
        """
        # =============================================================
        # ① 延迟 import MarkdownHeaderTextSplitter
        #    （还没装 langchain_text_splitters 时不崩，返回 None 即可）
        # =============================================================
        try:
            from langchain_text_splitters import MarkdownHeaderTextSplitter
        except Exception as e:
            print(f"[_markdown_header_split] WARN: import 失败，将跳过 Markdown 结构切分。原因：{e}")
            return None

        # =============================================================
        # ② 定义要切的 3 级标题
        #    元组的第一个元素是 Markdown 标记符号，第二个元素是我们给这级标题起的名字（会写进 metadata 里）
        #    这里只切到 3 级（# / ## / ###），#### 及更细的小标题不切（避免切得太碎）
        # =============================================================
        headers_to_split_on = [
            ("#",   "Header 1"),
            ("##",  "Header 2"),
            ("###", "Header 3"),
        ]

        # =============================================================
        # ③ 构造切分器实例
        #    strip_headers=False：标题行保留在段内容开头！
        #       原因：如果 strip_headers=True，段内容开头的「## 3.2 青光眼临床表现」会被单独丢掉，
        #       段正文直接从正文开始，丢失了"本段在讲什么主题"的关键上下文，RAG 检索效果会明显下降。
        #       我们明确保留标题行在 content 里，后续给 LLM 看 / 做向量都更准。
        # =============================================================
        splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=headers_to_split_on,
            strip_headers=False,
        )

        # =============================================================
        # ④ 执行切分
        #    MarkdownHeaderTextSplitter.split_text 返回 List[Document]：
        #       - Document.page_content 就是一段正文（含标题行，因为 strip_headers=False）
        #       - Document.metadata 里会自动按我们上面的 "Header 1/2/3" 填命中的标题链
        #         比如某段之前碰到过 # 第一章 > ## 1.2 眼球结构，
        #         metadata 就会是 {"Header 1": "第一章 眼科学基础", "Header 2": "1.2 眼球的结构"}
        # =============================================================
        try:
            split_raw: list = splitter.split_text(parent_doc.page_content)
        except Exception as e:
            # 任何切分异常（比如源 md 内容太怪）都不要让父流程崩，返回 None 交给 fallback
            print(f"[_markdown_header_split] WARN: split 失败，跳过 Markdown 切分。原因：{e}")
            return None

        # 切分返回了空列表（比如整个 md 没任何标题），也当失败处理，交给外层 fallback
        if not split_raw:
            return None

        # =============================================================
        # ⑤ 把 LangChain 返回的 List[Document] 统一包装成外层期待的 List[dict]
        #    好处：
        #      - 外层 chunk_documents 已经兼容了 List[dict] 形状，不需要改
        #      - "section" 字段额外做成人类可读的「标题路径字符串」，调试时一眼能看出这段属于哪一章哪一节
        # =============================================================
        result: list = []
        for item in split_raw:
            if not isinstance(item, Document):
                # 保险：不认识的元素直接跳过（理论上不会发生）
                continue

            raw_content = (item.page_content or "").strip()
            if not raw_content:
                # 空段（比如连续两个标题之间没正文）直接丢，避免噪声块
                continue

            # --- 从 item.metadata 里拼 section 字符串：# Header1 > ## Header2 > ### Header3
            # 例："# 第一章 眼科学基础 > ## 1.2 眼球的结构"
            section_parts = []
            im = item.metadata or {}
            if "Header 1" in im and im["Header 1"]:
                section_parts.append(f"# {im['Header 1']}")
            if "Header 2" in im and im["Header 2"]:
                section_parts.append(f"## {im['Header 2']}")
            if "Header 3" in im and im["Header 3"]:
                section_parts.append(f"### {im['Header 3']}")
            section_str = " > ".join(section_parts) if section_parts else None

            result.append({
                "content": raw_content,
                "section": section_str,
            })

        # 理论上不应该空，但还是兜底一次（比如上面全是空段），让外层走 fallback
        if not result:
            return None

        # --- 调试打印（可选：想看这本书切成了几段，就把下一行取消注释）
        # title = parent_doc.metadata.get("doc_title", "(no title)")
        # print(f"  [md_split] {title}: 产生 {len(result)} 个标题段")

        return result

    def get_parent_documents(self, child_chunks):
        """
        根据子块获取对应的父文档（小块检索 → 大块生成 的核心 O(1) 反查）

        入参：
            child_chunks: List[Document] —— 从检索阶段拿回来的 top_k 子块（通常 3~15 个）
        返回：
            List[Document] —— 子块对应的父文档（已去重，按「命中子块次数 hit_count」从高到低排序）
                            每个返回的父文档.metadata 会临时多一个字段 _hit_count，告诉 LLM 这份资料被命中了几次
        """
        # =============================================================
        # ① 兜底：child_chunks 为空直接返回空列表（没什么好查的）
        # =============================================================
        if not child_chunks:
            return []

        # =============================================================
        # ② 先从 self.documents 里建一个 parent_id → Document 的 O(1) 查找表
        #    （每次调这个方法都会建，但 documents 一般最多几百个，建起来 <0.1ms，成本几乎为 0）
        #    如果你特别在意性能，可以把这个表缓存成 self._id_to_doc 的实例变量，第一次调再 lazy build
        # =============================================================
        id_to_doc = {}
        for doc in self.documents:
            pid = doc.metadata.get("parent_id")
            if pid:
                id_to_doc[pid] = doc

        # =============================================================
        # ③ 遍历所有子块 → 统计每个父文档被命中了多少子块（hit_count）
        #    同时收集"命中过哪些父文档 parent_id 的唯一集合（去重用）"
        # =============================================================
        hit_count = {}   # {parent_id: 命中了几个子块}
        seen_parent_ids = set()   # 纯粹是一个保险集合

        for chunk in child_chunks:
            # 防御：入参里居然不是 Document（比如手误传了 dict/str），跳过不崩
            if not isinstance(chunk, Document):
                continue
            cm = chunk.metadata or {}
            cid = cm.get("chunk_id")           # 子块自己的 ID
            pid = cm.get("parent_id")          # 子块 metadata 里自带的 parent_id（优先用这个，更快）

            # --- 如果子块 metadata 里已经直接写了 parent_id（我们的 chunk_documents 每次都写了），直接用
            if pid is None:
                # --- 如果某个人手工造的子块没填 parent_id，就再走一次 parent_child_map[chunk_id] 反查
                if cid is None:
                    # 两个 ID 都没有？这个子块没法溯源，跳过
                    continue
                pid = self.parent_child_map.get(cid)

            # --- 还是找不到 parent_id（或者 parent_child_map 里压根没这条记录，数据脏了），跳过
            if pid is None or pid not in id_to_doc:
                continue

            # 记录命中 + 累加计数
            seen_parent_ids.add(pid)
            hit_count[pid] = hit_count.get(pid, 0) + 1

        # =============================================================
        # ④ 从 id_to_doc 里把命中的父文档捞出来，附上 _hit_count，按 hit_count 从高到低排序
        #    hit_count 相同的按什么排？→ 按 source_level A > B > C 排（权威教材优先），
        #    source_level 也相同就按父文档原顺序排（稳定排序），保证结果可复现
        # =============================================================
        level_priority = {"A": 0, "B": 1, "C": 2}  # 数字越小越排前面
        result_with_meta = []
        for pid in seen_parent_ids:
            parent_doc = id_to_doc[pid]   # 实际父文档对象
            hc = hit_count[pid]           # 这个父文档命中的子块数
            # 排序键 tuple：第 1 键 = -hit_count（要降序，所以取负数）；第 2 键 = source_level 优先级；第 3 键 = parent_id（纯为了可复现，完全相同的条件下稳定排序）
            sort_key = (
                -hc,
                level_priority.get(parent_doc.metadata.get("source_level", "C"), 3),
                pid,
            )
            result_with_meta.append((sort_key, hc, parent_doc))

        # 按 sort_key 排序
        result_with_meta.sort(key=lambda x: x[0])

        # =============================================================
        # ⑤ 输出：排序后把 _hit_count 写进父文档的 metadata（临时字段，不影响原 documents 的元数据，
        #    这里我们 .copy() 一份 metadata，防止改动 self.documents 里的原始对象）
        # =============================================================
        final_docs = []
        for _sort_key, hc, parent_doc in result_with_meta:
            # --- 拷贝 metadata（不污染原父文档的 metadata）
            new_meta = dict(parent_doc.metadata)
            new_meta["_hit_count"] = hc
            # --- 拷贝 Document（LangChain Document 有 copy 方法最好，没的话直接构造一个新的）
            try:
                new_doc = parent_doc.copy()
                new_doc.metadata = new_meta
            except Exception:
                # 兜底：老版本 LangChain Document 可能没有 copy 方法，直接新建
                new_doc = Document(page_content=parent_doc.page_content, metadata=new_meta)
            final_docs.append(new_doc)

        return final_docs

    def get_statistics(self) -> Dict:
        """
        获取数据统计信息（对外的"数字仪表盘"，初始化完成后调一次就能看到所有数据健康指标）
        返回一个扁平的 dict，包含：
          - 总量：父文档数 / 子块数 / 父子映射条目数
          - 父文档维度的分类分布：category / sub_category / source_level / doc_type / difficulty
          - 子块维度的分类分布：split_from（切块来源分布，观察 Markdown 切占比多少）
          - 块大小分布：子块字符数 min/max/mean/median + 几个百分位区间
          - 派生指标：平均每父文档多少子块 / 平均子块字符数
        """
        from collections import Counter

        # =============================================================
        # ① 总量：3 个数字直接拿
        # =============================================================
        n_docs = len(self.documents)
        n_chunks = len(self.chunks)
        n_map_entries = len(self.parent_child_map)

        stats = {
            # --- 总量类（一眼就能看 RAG 的数据集规模有多大）---
            "total_documents":   n_docs,        # 父文档总数（我们这边就是 38）
            "total_chunks":      n_chunks,      # 子块总数（~12136 左右）
            "total_map_entries": n_map_entries, # parent_child_map 条目数（必须 == n_chunks）
        }

        # =============================================================
        # ② 父文档维度的分类分布（category / sub_category / source_level / doc_type / difficulty）
        #    统一用 collections.Counter 数，一行一个维度，代码很干净
        # =============================================================
        if n_docs > 0:
            # --- 类别分布 ---
            stats["documents_by_category"]     = dict(Counter(d.metadata.get("category",     "unknown") for d in self.documents))
            stats["documents_by_sub_category"] = dict(Counter(d.metadata.get("sub_category", "unknown") for d in self.documents))
            stats["documents_by_source_level"] = dict(Counter(d.metadata.get("source_level", "C")       for d in self.documents))
            stats["documents_by_doc_type"]     = dict(Counter(d.metadata.get("doc_type",     "article") for d in self.documents))
            stats["documents_by_difficulty"]   = dict(Counter(d.metadata.get("difficulty",   "intermediate") for d in self.documents))
            # --- 派生指标：平均每本父文档 → 多少子块 ---
            stats["avg_chunks_per_document"] = round(n_chunks / n_docs, 2)
        else:
            # 兜底：还没调 load_documents 就调 get_statistics，也不要返回没字段
            for k in [
                "documents_by_category", "documents_by_sub_category",
                "documents_by_source_level", "documents_by_doc_type",
                "documents_by_difficulty", "avg_chunks_per_document",
            ]:
                stats[k] = {} if "by_" in k else 0

        # =============================================================
        # ③ 子块维度的分类分布（split_from 切块来源分布：观察 Markdown 切是不是生效了）
        # =============================================================
        if n_chunks > 0:
            stats["chunks_by_split_from"] = dict(Counter(c.metadata.get("split_from", "unknown") for c in self.chunks))

            # --- 子块字符数分布：基础统计 ---
            lens = sorted(len(c.page_content) for c in self.chunks)
            stats["chunk_length_min"]    = lens[0]
            stats["chunk_length_max"]    = lens[-1]
            stats["chunk_length_mean"]   = round(sum(lens) / len(lens), 2)
            stats["chunk_length_median"] = lens[len(lens) // 2]
            # --- 子块字符数分布：区间统计（看有多少落在理想 500~1500 区间）---
            def count_range(a, b):
                import bisect
                # bisect_left/right 找区间端点下标，差就是区间内数量（O(logN) 比遍历 O(N) 快）
                lo = bisect.bisect_left(lens, a)
                if b is None:
                    return len(lens) - lo
                hi = bisect.bisect_left(lens, b)
                return hi - lo
            stats["chunk_length_buckets"] = {
                "<100":      count_range(0, 100),
                "100_499":   count_range(100, 500),
                "500_1500":  count_range(500, 1501),
                "1501_1600": count_range(1501, 1601),
                ">1600":     count_range(1601, None),
            }
            # --- 派生指标：平均子块字符数 ---
            stats["avg_chunk_length"] = stats["chunk_length_mean"]
        else:
            stats["chunks_by_split_from"] = {}
            for k in [
                "chunk_length_min", "chunk_length_max",
                "chunk_length_mean", "chunk_length_median",
                "avg_chunk_length",
            ]:
                stats[k] = 0
            stats["chunk_length_buckets"] = {}

        # =============================================================
        # ④ 数据一致性校验（能直接在统计里看出 bug 的几个布尔标志，
        #    以后你想做健康检查直接看这几个 flag 就行）
        # =============================================================
        stats["_consistency_flags"] = {
            # 正常情况：每块都必须在 parent_child_map 有一条，多一条少一条都算数据脏了
            "chunks_eq_map_entries": n_chunks == n_map_entries,
            # 正常情况：平均每父文档子块数 ≥1（38 份父文档每份至少 1 块）
            "avg_chunks_ge_1": stats.get("avg_chunks_per_document", 0) >= 1,
            # 正常情况：子块长度 >1600 字符的数量应当 == 0（我们 CHUNK_MAX_CHARS=1500，即使重叠也不应该超太多）
            "no_too_long_chunks": stats.get("chunk_length_buckets", {}).get(">1600", 0) == 0,
        }

        return stats

    @staticmethod
    def get_supported_categories() -> List[str]:
        """
        获取支持的医疗文档分类（对外 API 契约：任何地方想做 metadata 过滤的分类白名单都走这个，
        不要在外面的模块自己手写 ["diseases", "health_articles"] 字符串数组，
        以后加/改分类只需要改这里一个地方。）
        """
        return [
            "diseases",         # 疾病/教材/图谱类（diseases/ophthalmology, diseases/optometry）
            "health_articles",  # 健康科普/职业资格/门诊笔记/Q&A
            "guidelines",       # 指南类（目前占位空目录，以后加了指南 md 会自动跑进来）
            "root",             # 兜底：data/ 根目录直接放的 md（不建议，但要有兼容值）
        ]

    @staticmethod
    def get_supported_difficulties() -> List[str]:
        """
        获取支持的阅读难度列表（同上，统一来源，别在外面硬编码字符串）
        顺序：由浅入深（beginner → intermediate → advanced）
        """
        return [
            "beginner",      # 入门：职业资格初级/基础知识 / 门诊笔记 / 科普 Q&A / 儿童科普
            "intermediate",  # 中级：一般教材（第2/3版专项书） / 职业资格中级/高级/技师
            "advanced",      # 高级：Wills 图谱 / 眼科学第十版 / 临床双眼视觉学 / 低视力学 / 屈光手术学 / 硬性角膜接触镜
        ]
