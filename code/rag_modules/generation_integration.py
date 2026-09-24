"""
生成集成模块 - 负责LLM集成和眼科/视光回答生成
"""

import os
from typing import List, Dict, Any

from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_openai import ChatOpenAI                       # AIHUBMIX 提供 OpenAI 兼容 API
from langchain_core.documents import Document
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser


class GenerationIntegrationModule:
    """生成集成模块 - 负责查询路由、查询重写和 5 路由眼科/视光回答生成"""

    def __init__(self, model_name: str = "glm-5.3-flash",
                 temperature: float = 0.3, max_tokens: int = 2048):
        # =============================================================
        # ① 参数记录（temperature 锁 0.3，外部传更高/更低也无效——已在 __init__ 默认值锁死）
        # =============================================================
        self.model_name = model_name
        self.temperature = 0.3              # 用户硬约束：锁死 0.3，忽略传入的其他值
        self.max_tokens = max_tokens
        self.llm = None

        # =============================================================
        # ② __init__ 末尾立刻调用 setup_llm()，避免用户忘记手动 init
        # =============================================================
        self.setup_llm()

    def setup_llm(self):
        """
        初始化大语言模型（用 AIHUBMIX 的 OpenAI 兼容接口）。
        幂等：self.llm 已存在则直接 return（避免重复 new 浪费资源）。
        """
        # =============================================================
        # ① 幂等保护：已经有 llm 对象就不重复初始化
        # =============================================================
        if self.llm is not None:
            return

        # =============================================================
        # ② 取 AIHUBMIX_API_KEY（用户 04_doc 指定的默认 LLM 提供商）
        # =============================================================
        api_key = os.getenv("AIHUBMIX_API_KEY")
        if not api_key:
            # 中文错误提示：告诉用户从 .env.example 复制并填 key
            print(
                "[ERROR][GenerationIntegrationModule.setup_llm] 未找到 AIHUBMIX_API_KEY 环境变量。\n"
                "         请从 code/.env.example 复制成 .env，填入你的 API Key；\n"
                "         或者手动在 shell 里 set AIHUBMIX_API_KEY=xxx 后再运行。"
            )
            raise ValueError(
                "[GenerationIntegrationModule.setup_llm] 缺少 AIHUBMIX_API_KEY 环境变量，"
                "无法实例化 glm-5.3-flash。请检查 code/.env 配置。"
            )

        # =============================================================
        # ③ 实例化 ChatOpenAI（AIHUBMIX 的 base_url = https://aihubmix.com/v1，完全兼容 OpenAI 协议）
        #    关键参数：
        #      - model = glm-5.3-flash（性价比高，练手够用）
        #      - temperature = 0.3（用户硬约束）
        #      - max_tokens = 2048（回答长度上限）
        # =============================================================
        self.llm = ChatOpenAI(
            model=self.model_name,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            api_key=api_key,
            base_url="https://aihubmix.com/v1",
        )

        print(f"[GenerationIntegrationModule.setup_llm] ✅ LLM 初始化成功：model={self.model_name}, temperature={self.temperature}")

    # =============================================================
    # 【通用工具方法】_build_context
    # =============================================================
    def _build_context(self, docs: List[Document], max_chars: int = 2000) -> str:
        """
        把命中的父文档列表拼成 LLM 能吃的「上下文字符串」。
        循环 docs → 每条拼元数据信息+文本 → 超 max_chars 截断。
        元数据用我们的 8 字段（doc_title/source_level/category/sub_category/doc_type/difficulty），
        不打印面向用户的提示，只做字符串拼接。
        """
        if not docs:
            return "（没有检索到相关参考资料）"

        context_parts: List[str] = []            # 每条父文档拼接后的字符串，最后用分隔线 join
        current_chars = 0                        # 已累计字符数（粗略估计上下文长度，防止超出 LLM context window）

        for i, doc in enumerate(docs, 1):
            meta = doc.metadata if isinstance(doc.metadata, dict) else {}

            # =============================================================
            # ① 拼元数据信息头（我们的 8 字段 schema，挑有用的展示，不堆噪声）
            # =============================================================
            header_parts = [f"【资料 {i}】"]
            if meta.get("doc_title"):
                header_parts.append(f'《{meta["doc_title"]}》')
            if meta.get("source_level"):                                     # A/B/C 等级，让 LLM 知道资料可信度
                header_parts.append(f" | 可信度等级: {meta['source_level']}")
            if meta.get("category") and meta.get("sub_category"):             # 分类：diseases/ophthalmology 这种
                header_parts.append(f" | 分类: {meta['category']}/{meta['sub_category']}")
            if meta.get("doc_type"):
                header_parts.append(f" | 类型: {meta['doc_type']}")
            if meta.get("difficulty"):
                header_parts.append(f" | 难度: {meta['difficulty']}")
            header_line = "".join(header_parts)

            # =============================================================
            # ② 拼完整文档块（头 + 正文）
            # =============================================================
            doc_text = f"{header_line}\n{doc.page_content.strip()}\n"

            # =============================================================
            # ③ 字符数超限就停止追加（后面的资料精度更低，截断不影响效果）
            # =============================================================
            if current_chars + len(doc_text) > max_chars:
                break
            context_parts.append(doc_text)
            current_chars += len(doc_text)

        # =============================================================
        # ④ 用 50 根等号的分隔线 join 每条资料，视觉上一眼能分开不同父文档
        # =============================================================
        divider = "\n" + "=" * 60 + "\n"
        return divider + divider.join(context_parts)

    # =============================================================
    # 【路由层 1/2】query_router
    # =============================================================
    def query_router(self, query: str) -> str:
        """
        查询路由 - 根据查询类型选择不同的处理方式

        Args:
            query: 用户查询

        Returns:
            路由类型 ('list', 'detail', 'general')
        """
        prompt = ChatPromptTemplate.from_template("""
根据用户的眼科/视光问题，将其分类为以下三种类型之一：

1. 'list' - 用户想要获取【疾病/异常/检查项目的列表或清单】，只需要列条目名称
   例如：眼科常见疾病有哪些、近视手术分几种、配镜要查哪些项目

2. 'detail' - 用户想要【某个具体主题的详细原理/症状/治疗/操作流程】
   例如：青光眼有什么症状、配眼镜完整流程、阿托品眼用凝胶用法

3. 'general' - 其他一般性科普/生活咨询问题
   例如：平时怎么保护眼睛、看手机多久休息、儿童近视能不能恢复

请只返回分类结果：list、detail 或 general

用户问题: {query}

分类结果:""")

        chain = (
            {"query": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )

        result = chain.invoke(query).strip().lower()

        # 确保返回有效的路由类型
        if result in ['list', 'detail', 'general']:
            return result
        else:
            return 'general'

    # =============================================================
    # 【路由层 2/2】query_rewrite
    # =============================================================
    def query_rewrite(self, query: str) -> str:
        """
        智能查询重写：模糊口语查询 → 补充眼科/视光专业术语，提高检索召回。
        使用「先分析 + 再重写」的 C8 同款双段结构，避免 LLM 在规则压力下输出空串。
        """
        prompt = ChatPromptTemplate.from_template("""
你是眼科/视光专业的查询优化助手。分析下面的用户查询，按照【分析】→【重写结果】两步输出。

原始查询: {query}

【判断标准】
- 具体明确（已有疾病名/药品名/专业术语）：不重写，原样输出。
- 模糊口语/过于宽泛：补充专业术语或限定词（如「症状」「治疗」「原因」「检查流程」）。
- 多个并列问题：合并成一句连贯的检索查询，不要拆分。
- 输出控制在 30 个汉字以内。

【示例】
- 「眼睛疼」 → 眼睛疼痛常见原因与症状鉴别
- 「孩子视力差」 → 儿童视力下降常见原因与检查流程
- 「配眼镜」 → 框架眼镜验光配镜完整流程
- 「青光眼怎么治」 → 青光眼怎么治（不重写）

请严格按以下格式输出，不要加多余内容：

**分析：**
- （1~2 行要点说明）

**重写结果：**
XXXX""")

        chain = (
            {"query": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )
        raw_output = chain.invoke(query).strip()

        # 从输出中提取「重写结果」之后的内容
        rewritten = query
        try:
            idx = raw_output.rfind("**重写结果：**")
            if idx == -1:
                idx = raw_output.rfind("重写结果：")
            if idx != -1:
                # 取冒号之后第一行，去掉前后空白/引号/反引号
                tail = raw_output[idx:].splitlines()[1:] if '\n' in raw_output[idx:] else []
                # 如果行内换行不够，用逗号切 tail 为第 0 段后的剩余
                if not tail and ':' in raw_output[idx:]:
                    after_colon = raw_output[idx:].split(':', 1)[1]
                    tail = [after_colon]
                if tail:
                    candidate = tail[0].strip().strip('"').strip("'").strip("`").strip()
                    if candidate:
                        rewritten = candidate
        except Exception:
            rewritten = query

        # 防御：结果为空或长度异常时回退原 query
        if not rewritten or len(rewritten.replace(' ', '')) < 2:
            rewritten = query

        # 打印重写记录（本项目风格：不用 logging，用 print）
        if rewritten != query:
            print(f"[GenerationIntegrationModule.query_rewrite] 重写: {query!r} → {rewritten!r}")
        else:
            print(f"[GenerationIntegrationModule.query_rewrite] 无需重写: {query!r}")
        return rewritten

    # =============================================================
    # 【生成方法 1/10】generate_list_answer
    # =============================================================
    def generate_list_answer(self, query: str, context_docs: List[Document]) -> str:
        """
        列表式回答：只给条目标题，不展开细节。
        空 context → 提示未找到；否则从 metadata 提取 doc_title 去重 → 简洁列表输出。
        用「命中的父文档 doc_title + 文档首行抓取到的条目名」做列表。
        """
        if not context_docs:
            return "抱歉，没有检索到相关的眼科/视光资料。"

        # =============================================================
        # ① 用 LCEL 链让 LLM 基于 context 提取列表（让 LLM 从资料正文里抽条目）
        # =============================================================
        context = self._build_context(context_docs)
        prompt = ChatPromptTemplate.from_template("""
你是眼科/视光专业助手。用户需要【清单/列表形式】的回答。

用户问题: {question}

参考资料:
{context}

请提供清晰、实用的回答，建议按以下方式组织（可根据实际内容灵活调整）：
- 开头用一句话自然说明这份清单包含什么内容
- 主体部分用编号列表展示条目，每条可用一句简短介绍帮助理解
- 如果原文中有明确的分类，可以按类别分组；没有就直接平铺
- 参考资料中没有的信息诚实说明，不要为了凑数瞎编

回答:""")
        chain = (
            {"question": RunnablePassthrough(), "context": lambda _: context}
            | prompt
            | self.llm
            | StrOutputParser()
        )
        return chain.invoke(query)

    def generate_list_answer_stream(self, query: str, context_docs: List[Document]):
        """generate_list_answer 的流式版本：把上面最后一步 invoke → chain.stream"""
        if not context_docs:
            yield "抱歉，没有检索到相关的眼科/视光资料。"
            return
        context = self._build_context(context_docs)
        prompt = ChatPromptTemplate.from_template("""
你是眼科/视光专业助手。用户需要【清单/列表形式】的回答。

用户问题: {question}

参考资料:
{context}

请提供清晰、实用的回答，建议按以下方式组织（可根据实际内容灵活调整）：
- 开头用一句话自然说明这份清单包含什么内容
- 主体部分用编号列表展示条目，每条可用一句简短介绍帮助理解
- 如果原文中有明确的分类，可以按类别分组；没有就直接平铺
- 参考资料中没有的信息诚实说明，不要为了凑数瞎编

回答:""")
        chain = (
            {"question": RunnablePassthrough(), "context": lambda _: context}
            | prompt
            | self.llm
            | StrOutputParser()
        )
        for chunk in chain.stream(query):
            yield chunk

    # =============================================================
    # 【生成方法 3/10】generate_step_by_step_answer = disease_detail 详细解释
    # =============================================================
    def generate_step_by_step_answer(self, query: str, context_docs: List[Document]) -> str:
        """
        详细解释型回答（对应路由 disease_detail）：原理/症状/病因/分类 这种偏知识讲解类。
        结构化模板（5 个眼科专用小节，灵活可省略）。
        """
        if not context_docs:
            return "抱歉，没有检索到相关的参考资料，无法给出准确解释。"
        context = self._build_context(context_docs)
        prompt = ChatPromptTemplate.from_template("""
你是眼科/视光专业助手。请根据参考资料，为用户的问题提供**结构化、分小节**的详细解释。

用户问题: {question}

参考资料:
{context}

请灵活组织回答，建议包含以下部分（可根据实际内容调整，某部分没信息就完全省略，不要硬凑）：

## 📖 概述
用自然语言简要介绍这个主题的定义、特点或核心概念。

## 🔍 常见原因 / 发病机制
分点列出主要原因或发病机制说明。

## 🩺 典型症状 / 临床表现
分点列出典型症状或表现，优先使用原文中的症状描述。

## 📂 分类 / 分型 / 分度
有明确分类时列出，没有就整个小节省略。

## 💡 相关注意 / 临床要点 / 常见误区
当资料里有明确的注意事项、误区说明或实用要点时列出；没有就省略。

根据实际内容灵活调整结构，不要强行填充缺失的部分。
重点突出实用性，必要时可以用表格整理对比信息。
参考资料中没有的信息诚实说明未提及；语言自然流畅，不要机械堆砌原文。

回答:""")
        chain = (
            {"question": RunnablePassthrough(), "context": lambda _: context}
            | prompt
            | self.llm
            | StrOutputParser()
        )
        return chain.invoke(query)

    def generate_step_by_step_answer_stream(self, query: str, context_docs: List[Document]):
        """generate_step_by_step_answer 流式版本：invoke → for chunk in .stream yield"""
        if not context_docs:
            yield "抱歉，没有检索到相关的参考资料，无法给出准确解释。"
            return
        context = self._build_context(context_docs)
        prompt = ChatPromptTemplate.from_template("""
你是眼科/视光专业助手。请根据参考资料，为用户的问题提供**结构化、分小节**的详细解释。

用户问题: {question}

参考资料:
{context}

请灵活组织回答，建议包含以下部分（可根据实际内容调整，某部分没信息就完全省略，不要硬凑）：

## 📖 概述
用自然语言简要介绍这个主题的定义、特点或核心概念。

## 🔍 常见原因 / 发病机制
分点列出主要原因或发病机制说明。

## 🩺 典型症状 / 临床表现
分点列出典型症状或表现，优先使用原文中的症状描述。

## 📂 分类 / 分型 / 分度
有明确分类时列出，没有就整个小节省略。

## 💡 相关注意 / 临床要点 / 常见误区
当资料里有明确的注意事项、误区说明或实用要点时列出；没有就省略。

根据实际内容灵活调整结构，不要强行填充缺失的部分。
重点突出实用性，必要时可以用表格整理对比信息。
参考资料中没有的信息诚实说明未提及；语言自然流畅，不要机械堆砌原文。

回答:""")
        chain = (
            {"question": RunnablePassthrough(), "context": lambda _: context}
            | prompt
            | self.llm
            | StrOutputParser()
        )
        for chunk in chain.stream(query):
            yield chunk

    # =============================================================
    # 【生成方法 5/10】generate_drug_answer = 药品/眼药水说明书式 8 项输出
    # =============================================================
    def generate_drug_answer(self, query: str, context_docs: List[Document]) -> str:
        """
        药品/眼药水说明书式回答（对应路由 drug_info）。
        注意：我们目前 38 份资料里几乎没有药品专项数据，所以空 context 概率高，返回诚实提示；
        如果命中了资料（比如眼科学某章里提到了某药），就按 8 项结构输出。
        结构化模板风格（分 ## 小节 + 每节规则）。
        """
        if not context_docs:
            return "抱歉，当前参考资料中未检索到该药品/眼药水的专项说明，无法给出准确用法信息。"
        context = self._build_context(context_docs)
        prompt = ChatPromptTemplate.from_template("""
你是眼科/视光专业助手。用户询问药品/眼药水信息，请根据参考资料输出实用的结构化回答。

用户问题: {question}

参考资料:
{context}

建议包含以下部分（可根据实际资料灵活调整，没信息就整项省略，不要硬写【未提及】占位）：

## 1. 药品名称
通用名与商品名。

## 2. 成分 / 主要作用机制
活性成分、药理作用类别。

## 3. 适应症
原文描述的适用症状与疾病。

## 4. 用法用量
给药途径、频次、疗程；滴眼液请说明每次几滴、每日几次。

## 5. 不良反应 / 副作用
原文列出的可能副作用，分点展示。

## 6. 禁忌症
明确不能使用的人群或情况。

## 7. 注意事项
孕妇、儿童、老年人、驾驶、戴隐形眼镜时的特殊注意。

## 8. 资料来源
列出命中的参考资料 doc_title，书名号包裹。

根据实际内容灵活组织，不要强行填充没有的信息。
重点突出实用性；可以用表格展示对比项（如不同浓度的区别）。
原文中有特殊说明时，可加「💡 小贴士」小节。
参考资料中没有的信息诚实说明，不要编造批准文号、生产厂家等教科书没有的内容。

回答:""")
        chain = (
            {"question": RunnablePassthrough(), "context": lambda _: context}
            | prompt
            | self.llm
            | StrOutputParser()
        )
        return chain.invoke(query)

    def generate_drug_answer_stream(self, query: str, context_docs: List[Document]):
        """generate_drug_answer 流式版本"""
        if not context_docs:
            yield "抱歉，当前参考资料中未检索到该药品/眼药水的专项说明，无法给出准确用法信息。"
            return
        context = self._build_context(context_docs)
        prompt = ChatPromptTemplate.from_template("""
你是眼科/视光专业助手。用户询问药品/眼药水信息，请根据参考资料输出实用的结构化回答。

用户问题: {question}

参考资料:
{context}

建议包含以下部分（可根据实际资料灵活调整，没信息就整项省略，不要硬写【未提及】占位）：

## 1. 药品名称
通用名与商品名。

## 2. 成分 / 主要作用机制
活性成分、药理作用类别。

## 3. 适应症
原文描述的适用症状与疾病。

## 4. 用法用量
给药途径、频次、疗程；滴眼液请说明每次几滴、每日几次。

## 5. 不良反应 / 副作用
原文列出的可能副作用，分点展示。

## 6. 禁忌症
明确不能使用的人群或情况。

## 7. 注意事项
孕妇、儿童、老年人、驾驶、戴隐形眼镜时的特殊注意。

## 8. 资料来源
列出命中的参考资料 doc_title，书名号包裹。

根据实际内容灵活组织，不要强行填充没有的信息。
重点突出实用性；可以用表格展示对比项（如不同浓度的区别）。
原文中有特殊说明时，可加「💡 小贴士」小节。
参考资料中没有的信息诚实说明，不要编造批准文号、生产厂家等教科书没有的内容。

回答:""")
        chain = (
            {"question": RunnablePassthrough(), "context": lambda _: context}
            | prompt
            | self.llm
            | StrOutputParser()
        )
        for chunk in chain.stream(query):
            yield chunk

    # =============================================================
    # 【生成方法 7/10】generate_treatment_answer = 流程/步骤式回答
    # =============================================================
    def generate_treatment_answer(self, query: str, context_docs: List[Document]) -> str:
        """
        操作流程/步骤式回答（对应路由 treatment_info）：配眼镜流程、验光步骤、怎么戴隐形眼镜这种。
        分步骤模板（📝👣⚠️✅ 4 段眼科专用结构）。
        """
        if not context_docs:
            return "抱歉，参考资料中未找到对应操作流程的说明。"
        context = self._build_context(context_docs)
        prompt = ChatPromptTemplate.from_template("""
你是眼科/视光专业助手。请根据参考资料，为用户的问题提供**分步骤、可操作**的指导流程。

用户问题: {question}

参考资料:
{context}

请灵活组织回答，建议包含以下部分（可根据实际资料灵活调整，不硬凑）：

## 📝 适用场景 / 前置条件
简要说明这个流程适用的情况，以及做之前需要准备什么物品或环境。

## 👣 详细操作步骤
用编号 1. 2. 3. … 分步骤说明，每一步写清楚做什么、怎么做，以及判断标准或注意点。

## ⚠️ 常见错误 / 风险点
操作中容易出错的地方、可能出现的问题及处理方法；资料里没有就整项省略。

## ✅ 完成标准 / 后续评估
怎么判断流程做对了、做完后要观察什么、什么时候需要进一步处理；资料里没有就整项省略。

根据实际内容灵活调整结构，不要强行填充缺失的部分。
重点突出实用性与可操作性；步骤内容可以用表格整理对比项。
原文中有实用技巧或关键提醒时，可加「💡 小贴士」小节；没有就省略。
参考资料中没有的信息诚实说明未提及；语言自然流畅，不要机械堆砌原文。

回答:""")
        chain = (
            {"question": RunnablePassthrough(), "context": lambda _: context}
            | prompt
            | self.llm
            | StrOutputParser()
        )
        return chain.invoke(query)

    def generate_treatment_answer_stream(self, query: str, context_docs: List[Document]):
        """generate_treatment_answer 流式版本"""
        if not context_docs:
            yield "抱歉，参考资料中未找到对应操作流程的说明。"
            return
        context = self._build_context(context_docs)
        prompt = ChatPromptTemplate.from_template("""
你是眼科/视光专业助手。请根据参考资料，为用户的问题提供**分步骤、可操作**的指导流程。

用户问题: {question}

参考资料:
{context}

请灵活组织回答，建议包含以下部分（可根据实际资料灵活调整，不硬凑）：

## 📝 适用场景 / 前置条件
简要说明这个流程适用的情况，以及做之前需要准备什么物品或环境。

## 👣 详细操作步骤
用编号 1. 2. 3. … 分步骤说明，每一步写清楚做什么、怎么做，以及判断标准或注意点。

## ⚠️ 常见错误 / 风险点
操作中容易出错的地方、可能出现的问题及处理方法；资料里没有就整项省略。

## ✅ 完成标准 / 后续评估
怎么判断流程做对了、做完后要观察什么、什么时候需要进一步处理；资料里没有就整项省略。

根据实际内容灵活调整结构，不要强行填充缺失的部分。
重点突出实用性与可操作性；步骤内容可以用表格整理对比项。
原文中有实用技巧或关键提醒时，可加「💡 小贴士」小节；没有就省略。
参考资料中没有的信息诚实说明未提及；语言自然流畅，不要机械堆砌原文。

回答:""")
        chain = (
            {"question": RunnablePassthrough(), "context": lambda _: context}
            | prompt
            | self.llm
            | StrOutputParser()
        )
        for chunk in chain.stream(query):
            yield chunk

    # =============================================================
    # 【生成方法 9/10】generate_basic_answer = 兜底通用科普
    # =============================================================
    def generate_basic_answer(self, query: str, context_docs: List[Document]) -> str:
        """
        通用型基础回答（对应 routing general）：兜底路由，问题不明确/生活科普类走这里。
        简单 prompt，不做强结构，LCEL 链 question + context 注入。
        """
        if not context_docs:
            return "参考资料中未找到相关内容，建议换更具体的关键词重新提问（如加具体疾病名/场景限定词）。"
        context = self._build_context(context_docs)
        prompt = ChatPromptTemplate.from_template("""
你是眼科/视光专业助手。请根据下面的参考资料回答用户的问题。

用户问题: {question}

参考资料:
{context}

请提供详细、实用的回答。可根据实际内容灵活组织结构：
- 开头先用一句话自然说明核心内容
- 需要分点的地方用编号或项目符号；适合对比的信息可以用表格整理
- 原文中有实用技巧或小知识时，可加「💡 小贴士」「小知识」小节
- 参考资料中信息不足时诚实说明，不要为了凑字数编造内容
- 语言自然流畅，重点内容可以加粗

回答:""")
        chain = (
            {"question": RunnablePassthrough(), "context": lambda _: context}
            | prompt
            | self.llm
            | StrOutputParser()
        )
        return chain.invoke(query)

    def generate_basic_answer_stream(self, query: str, context_docs: List[Document]):
        """generate_basic_answer 流式版本：chain.stream 循环 yield chunk"""
        if not context_docs:
            yield "参考资料中未找到相关内容，建议换更具体的关键词重新提问（如加具体疾病名/场景限定词）。"
            return
        context = self._build_context(context_docs)
        prompt = ChatPromptTemplate.from_template("""
你是眼科/视光专业助手。请根据下面的参考资料回答用户的问题。

用户问题: {question}

参考资料:
{context}

请提供详细、实用的回答。可根据实际内容灵活组织结构：
- 开头先用一句话自然说明核心内容
- 需要分点的地方用编号或项目符号；适合对比的信息可以用表格整理
- 原文中有实用技巧或小知识时，可加「💡 小贴士」「小知识」小节
- 参考资料中信息不足时诚实说明，不要为了凑字数编造内容
- 语言自然流畅，重点内容可以加粗

回答:""")
        chain = (
            {"question": RunnablePassthrough(), "context": lambda _: context}
            | prompt
            | self.llm
            | StrOutputParser()
        )
        for chunk in chain.stream(query):
            yield chunk
