"""
Medical RAG系统主程序
"""

import os
import sys
import logging
import time
from datetime import datetime
from pathlib import Path

# 添加模块路径
sys.path.append(str(Path(__file__).parent))

from dotenv import load_dotenv
from config import DEFAULT_CONFIG, MedicalRAGConfig
from rag_modules import (
    DataPreparationModule,
    IndexConstructionModule,
    RetrievalOptimizationModule,
    GenerationIntegrationModule,
)

# 加载环境变量
load_dotenv()

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MedicalRAGSystem:
    """眼科/视光 RAG 系统主类"""

    def __init__(self, config: MedicalRAGConfig = None):
        """
        初始化RAG系统

        Args:
            config: RAG系统配置，默认使用DEFAULT_CONFIG
        """
        self.config = config or DEFAULT_CONFIG
        self.data_module = None
        self.index_module = None
        self.retrieval_module = None
        self.generation_module = None

        # 检查数据路径
        if not Path(self.config.data_path).exists():
            raise FileNotFoundError(f"数据路径不存在: {self.config.data_path}")

        # 检查API密钥
        if not os.getenv("AIHUBMIX_API_KEY"):
            raise ValueError("请设置 AIHUBMIX_API_KEY 环境变量")

    def initialize_system(self):
        """初始化所有模块"""
        print("🚀 正在初始化RAG系统...")

        # 1. 初始化数据准备模块
        print("初始化数据准备模块...")
        self.data_module = DataPreparationModule(self.config.data_path)

        # 2. 初始化索引构建模块
        print("初始化索引构建模块...")
        self.index_module = IndexConstructionModule(
            model_name=self.config.embedding_model,
            index_save_path=self.config.index_save_path
        )

        # 3. 初始化生成集成模块
        print("🤖 初始化生成集成模块...")
        self.generation_module = GenerationIntegrationModule(
            model_name=self.config.llm_model,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens
        )

        print("✅ 系统初始化完成！")

    def build_knowledge_base(self):
        """构建知识库"""
        print("\n正在构建知识库...")

        # 1. 尝试加载已保存的索引
        vectorstore = self.index_module.load_index()

        # 旧版索引使用随机 parent_id/chunk_id，不能与当前加载的父文档正确关联。
        # 检测到旧格式时忽略它，并在下面按当前稳定 ID 方案重新构建、覆盖保存。
        if vectorstore is not None:
            indexed_docs = vectorstore.docstore._dict.values()
            if not all(
                doc.metadata.get("id_scheme") == "stable-path-chunk-v1"
                for doc in indexed_docs
            ):
                print("检测到旧版随机 ID 索引，将按当前文档重新构建索引...")
                vectorstore = None

        if vectorstore is not None:
            print("✅ 成功加载已保存的向量索引！")
            # 仍需要加载文档和分块用于检索模块
            print("加载文档...")
            self.data_module.load_documents()
            print("进行文本分块...")
            chunks = self.data_module.chunk_documents()
        else:
            print("未找到已保存的索引，开始构建新索引...")

            # 2. 加载文档
            print("加载文档...")
            self.data_module.load_documents()

            # 3. 文本分块
            print("进行文本分块...")
            chunks = self.data_module.chunk_documents()

            # 4. 构建向量索引
            print("构建向量索引...")
            vectorstore = self.index_module.build_vector_index(chunks)

            # 5. 保存索引
            print("保存向量索引...")
            self.index_module.save_index()

        # 6. 初始化检索优化模块
        print("初始化检索优化...")
        self.retrieval_module = RetrievalOptimizationModule(vectorstore, chunks)

        # 7. 显示统计信息
        stats = self.data_module.get_statistics()
        print(f"\n📊 知识库统计:")
        print(f"   文档总数: {stats['total_documents']}")
        print(f"   文本块数: {stats['total_chunks']}")
        print(f"   文档分类: {list(stats['documents_by_category'].keys())}")
        print(f"   难度分布: {stats.get('documents_by_difficulty', {})}")
        print(f"   切块方式: {dict(stats['chunks_by_split_from'])}")

        print("✅ 知识库构建完成！")

    def ask_question(self, question: str, stream: bool = False):
        """
        回答用户问题

        Args:
            question: 用户问题
            stream: 是否使用流式输出

        Returns:
            生成的回答或生成器
        """
        if not all([self.retrieval_module, self.generation_module]):
            raise ValueError("请先构建知识库")

        print(f"\n❓ 用户问题: {question}")

        # 1. 查询路由：3 分类 list/detail/general，返回值已是合法 3 类，非法兜底 general
        raw_route = self.generation_module.query_router(question)
        route_type = raw_route if raw_route in ("list", "detail", "general") else "general"
        print(f"🎯 查询类型: {route_type}")

        # 2. 智能查询重写（根据路由类型）
        if route_type == 'list':
            # 列表查询保持原查询
            rewritten_query = question
            print(f"📝 列表查询保持原样: {question}")
        else:
            # 详细查询和一般查询使用智能重写
            print("🤖 智能分析查询...")
            rewritten_query = self.generation_module.query_rewrite(question)

        # 3. 检索相关子块（自动应用元数据过滤）
        print("🔍 检索相关文档...")
        filters = self._extract_filters_from_query(question)
        if filters:
            print(f"应用过滤条件: {filters}")
            relevant_chunks = self.retrieval_module.metadata_filtered_search(
                rewritten_query, filters=filters, top_k=self.config.top_k
            )
        else:
            relevant_chunks = self.retrieval_module.hybrid_search(
                rewritten_query, top_k=self.config.top_k
            )

        # 显示检索到的子块信息（doc_title + section 或 内容片段）
        if relevant_chunks:
            chunk_info = []
            for chunk in relevant_chunks:
                meta = chunk.metadata if isinstance(chunk.metadata, dict) else {}
                doc_title = meta.get('doc_title', '未知文档')
                section = meta.get('section', '')
                if section:
                    chunk_info.append(f"{doc_title}({section})")
                else:
                    # 尝试从内容中提取章节标题（# 开头取第一行）
                    content_preview = chunk.page_content[:100].strip()
                    if content_preview.startswith('#'):
                        title_end = content_preview.find('\n') if '\n' in content_preview else len(content_preview)
                        section_title = content_preview[:title_end].replace('#', '').strip()
                        chunk_info.append(f"{doc_title}({section_title})")
                    else:
                        chunk_info.append(f"{doc_title}(内容片段)")
            print(f"找到 {len(relevant_chunks)} 个相关文档块: {', '.join(chunk_info)}")
        else:
            print(f"找到 {len(relevant_chunks)} 个相关文档块")

        # 4. 检查是否找到相关内容
        if not relevant_chunks:
            return "抱歉，没有找到相关的眼科/视光资料。请尝试其他关键词或提问方式。"

        # 5. 根据路由类型选择回答方式
        if route_type == 'list':
            # 列表查询：直接返回列表回答
            print("📋 生成[疾病/项目]列表...")
            relevant_docs = self.data_module.get_parent_documents(relevant_chunks)

            # 显示找到的文档名称
            doc_names = []
            for doc in relevant_docs:
                meta = doc.metadata if isinstance(doc.metadata, dict) else {}
                title = meta.get('doc_title', '未知文档')
                doc_names.append(title)

            if doc_names:
                print(f"找到文档: {', '.join(doc_names)}")

            if stream:
                return self.generation_module.generate_list_answer_stream(question, relevant_docs)
            else:
                return self.generation_module.generate_list_answer(question, relevant_docs)
        else:
            # 详细查询：获取完整文档并生成详细回答
            print("获取完整文档...")
            relevant_docs = self.data_module.get_parent_documents(relevant_chunks)

            # 显示找到的文档名称
            doc_names = []
            for doc in relevant_docs:
                meta = doc.metadata if isinstance(doc.metadata, dict) else {}
                title = meta.get('doc_title', '未知文档')
                doc_names.append(title)

            if doc_names:
                print(f"找到文档: {', '.join(doc_names)}")
            else:
                print(f"对应 {len(relevant_docs)} 个完整文档")

            print("✍️ 生成详细回答...")

            # 根据路由类型自动选择回答模式
            if route_type == "detail":
                # 详细查询使用分步指导模式
                if stream:
                    return self.generation_module.generate_step_by_step_answer_stream(question, relevant_docs)
                else:
                    return self.generation_module.generate_step_by_step_answer(question, relevant_docs)
            else:
                # 一般查询使用基础回答模式
                if stream:
                    return self.generation_module.generate_basic_answer_stream(question, relevant_docs)
                else:
                    return self.generation_module.generate_basic_answer(question, relevant_docs)

    def _extract_filters_from_query(self, query: str) -> dict:
        """
        从用户问题中提取元数据过滤条件
        """
        filters = {}
        # 分类关键词
        category_keywords = DataPreparationModule.get_supported_categories()
        for cat in category_keywords:
            if cat in query:
                filters['category'] = cat
                break

        # 难度关键词
        difficulty_keywords = DataPreparationModule.get_supported_difficulties()
        for diff in sorted(difficulty_keywords, key=len, reverse=True):
            if diff in query:
                filters['difficulty'] = diff
                break

        return filters

    def run_interactive(self):
        """运行交互式问答"""
        print("=" * 60)
        print("  👁️  Medical RAG - 眼科/视光智能问答  👁️")
        print("=" * 60)

        # 初始化系统
        self.initialize_system()

        # 构建知识库
        self.build_knowledge_base()

        print("\n交互式问答 (输入'退出'结束):")

        while True:
            try:
                user_input = input("\n您的问题: ").strip()
                if user_input.lower() in ['退出', 'quit', 'exit', '']:
                    break

                # 询问是否使用流式输出
                stream_choice = input("是否使用流式输出? (y/n, 默认y): ").strip().lower()
                use_stream = stream_choice != 'n'

                # ------- 计时开始 -------
                t_start = time.perf_counter()
                ts_start = datetime.now().strftime("%H:%M:%S.%f")[:-3]
                print(f"⏱️ [计时·开始] {ts_start}")

                print("\n回答:")
                if use_stream:
                    # 流式输出：捕获第一个 chunk 到达视为「开始输出正式回答」，此时结束计时
                    stream_iter = self.ask_question(user_input, stream=True)
                    first_chunk = True
                    for chunk in stream_iter:
                        if chunk is None:
                            continue
                        if first_chunk:
                            t_end = time.perf_counter()
                            ts_end = datetime.now().strftime("%H:%M:%S.%f")[:-3]
                            elapsed = t_end - t_start
                            print(f"⏱️ [计时·结束] {ts_end}  总耗时: {elapsed:.2f} 秒")
                            print(f"⏱️ 耗时：{elapsed:.2f} 秒")
                            first_chunk = False
                        print(chunk, end="", flush=True)
                    # 兜底：空回答/异常时也结束计时
                    if first_chunk:
                        t_end = time.perf_counter()
                        ts_end = datetime.now().strftime("%H:%M:%S.%f")[:-3]
                        elapsed = t_end - t_start
                        print(f"⏱️ [计时·结束] {ts_end}  总耗时: {elapsed:.2f} 秒")
                        print(f"⏱️ 耗时：{elapsed:.2f} 秒")
                    print("\n")
                else:
                    # 普通输出：ask_question 返回即视为「回答准备完毕」，此时结束计时
                    answer = self.ask_question(user_input, stream=False)
                    t_end = time.perf_counter()
                    ts_end = datetime.now().strftime("%H:%M:%S.%f")[:-3]
                    elapsed = t_end - t_start
                    print(f"⏱️ [计时·结束] {ts_end}  总耗时: {elapsed:.2f} 秒")
                    print(f"⏱️ 耗时：{elapsed:.2f} 秒")
                    print(f"{answer}\n")

            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"处理问题时出错: {e}")

        print("\n感谢使用 Medical RAG 系统！")


def main():
    """主函数"""
    try:
        # 创建RAG系统
        rag_system = MedicalRAGSystem()

        # 运行交互式问答
        rag_system.run_interactive()

    except Exception as e:
        logger.error(f"系统运行出错: {e}")
        print(f"系统错误: {e}")


if __name__ == "__main__":
    main()
