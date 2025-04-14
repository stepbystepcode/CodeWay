# src/rag.py

import os
import asyncio
import concurrent.futures
import hashlib
import json
from functools import partial
import pathlib
import shutil # 引入 shutil 用于删除目录
from dotenv import load_dotenv
from llama_index.core import (
    Settings,
    VectorStoreIndex,
    StorageContext,
    load_index_from_storage,
    # ServiceContext, # 引入 ServiceContext
)
from llama_index.readers.github import GithubRepositoryReader, GithubClient
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from datetime import datetime # 引入 datetime
import time
import traceback # 用于打印详细错误信息

# --- 配置和初始化 ---
current_dir = pathlib.Path(__file__).parent.absolute()
backend_dir = current_dir.parent
dotenv_path = backend_dir / ".env"
load_dotenv(dotenv_path=dotenv_path)

CACHE_DIR = backend_dir / "cache" / "github_indexes"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# 使用稍微少一点的 worker，避免 huggingface/tokenizers 的 fork 问题，虽然不一定能完全解决
# 你也可以尝试设置环境变量 TOKENIZERS_PARALLELISM=false
_executor = concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count() // 2 or 1)

# --- GitHubQAManager ---
class GitHubQAManager:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(GitHubQAManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self.qa_systems = {}
        self._initialized = True
        print("[GitHub索引] GitHub QA系统管理器已初始化") # 只打印一次

    def get_or_create_qa_system(self, owner: str, repo: str, docs_folder_path: str, mode: str = "local", branch: str = "main", use_cache: bool = True) -> "GitHubDocsQA | None":
        """
        获取或创建指定仓库的QA系统实例。
        如果创建失败，返回 None。
        """
        key = f"{owner}/{repo}/{docs_folder_path}" # 使用 / 分隔更符合路径习惯
        print(f"[GitHub索引] 获取到的参数: owner={owner}, repo={repo}, branch={branch}, docs_folder_path='{docs_folder_path}'")

        if key not in self.qa_systems:
            print(f"[GitHub索引] 开始为 {key} 创建新的GitHub QA系统")
            print(f"[GitHub索引] 配置信息: 分支 = {branch}, 模式 = {mode}, 使用缓存 = {use_cache}")
            try:
                # 创建实例
                new_qa_system = GitHubDocsQA(
                    owner=owner,
                    repo=repo,
                    branch=branch,
                    docs_folder_path=docs_folder_path,
                    mode=mode,
                    use_cache=use_cache
                )
                # 检查初始化是否成功 (index 和 query_engine 是否有效)
                if new_qa_system.index is None or new_qa_system.query_engine is None:
                     print(f"[GitHub索引] 警告: GitHub QA系统 {key} 未能成功初始化索引或查询引擎。请检查日志。")
                     # 不存储失败的实例
                     return None
                else:
                    self.qa_systems[key] = new_qa_system
                    print(f"[GitHub索引] GitHub QA系统创建完成: {key}")

            except Exception as e:
                 print(f"[GitHub索引] 创建 QA 系统 {key} 时发生严重错误: {e}")
                 traceback.print_exc()
                 # 确保不会存储失败的实例
                 if key in self.qa_systems:
                     del self.qa_systems[key]
                 return None # 指示创建失败
        else:
            print(f"[GitHub索引] 使用已存在的GitHub QA系统: {key}")
            # 可选：检查已存在的系统是否仍然有效
            existing_system = self.qa_systems.get(key)
            if existing_system and (existing_system.index is None or existing_system.query_engine is None):
                 print(f"[GitHub索引] 警告: 已存在的 QA 系统 {key} 状态无效。将尝试重新创建。")
                 del self.qa_systems[key] # 删除无效实例
                 # 递归调用自身来重新创建，注意避免无限递归（虽然理论上不会）
                 return self.get_or_create_qa_system(owner, repo, docs_folder_path, mode, branch, use_cache)


        return self.qa_systems.get(key) # 使用 get 以防万一

    def list_repositories(self):
        return list(self.qa_systems.keys())

# 创建全局实例
github_qa_manager = GitHubQAManager()

# --- GitHubDocsQA ---
class GitHubDocsQA:
    def __init__(
        self,
        owner: str = "stepbystepcode",
        repo: str = "CodeWay",
        branch: str = "main",
        docs_folder_path: str = "", # 默认为空字符串，表示根目录
        mode: str = "local",
        use_cache: bool = True
    ):
        """Initialize the GitHub documentation QA system."""
        self.owner = owner
        self.repo = repo
        self.branch = branch
        # 标准化 docs_folder_path，去除首尾斜杠
        self.docs_folder_path = docs_folder_path.strip('/')
        self.mode = mode
        self.use_cache = use_cache

        self.documents = None
        self.reader = None
        self.query_engine = None
        self.index = None
        self.embedding_model = None # 实例变量来存储模型
        self.cache_key = self._generate_cache_key() # 移到这里，确保 docs_folder_path 已处理
        self.cache_dir = CACHE_DIR / self.cache_key

        print(f"[GitHub索引] 初始化 QA 系统实例: {self.cache_key}")

        # *** 关键改动 1: 早期初始化和设置 Settings.embed_model ***
        try:
            self._initialize_embedding_model()
            if not self.embedding_model:
                 # 如果初始化失败，抛出异常或记录错误，防止后续操作
                 raise ValueError("[GitHub索引] 致命错误: 无法初始化嵌入模型。")

            # 设置全局 Settings (非常重要，需要在加载/创建索引前完成)
            # 注意：这会影响所有后续的 LlamaIndex 操作，要小心多线程/多实例冲突
            # 在这个单例管理器模式下问题不大，但在其他场景需谨慎
            Settings.embed_model = self.embedding_model
            print(f"[GitHub索引] 全局嵌入模型已设置为: {self.embedding_model.model_name}")

        except Exception as e:
            print(f"[GitHub索引] 初始化嵌入模型或设置全局Settings时出错: {e}")
            traceback.print_exc()
            # 标记为初始化失败，阻止后续操作
            self.index = None
            self.query_engine = None
            raise # 重新抛出异常，让上层知道初始化失败

        # 尝试从缓存加载
        loaded_from_cache = False
        if use_cache:
            try:
                loaded_from_cache = self._load_from_cache()
                if loaded_from_cache:
                    print(f"[GitHub索引] 已成功从本地缓存加载索引: {self.cache_key}")
            except Exception as e:
                 print(f"[GitHub索引] 尝试从缓存加载时发生错误: {e}。将尝试重新创建索引。")
                 traceback.print_exc()
                 loaded_from_cache = False # 确保标记为未加载
                 # 可选：删除可能损坏的缓存
                 if self.cache_dir.exists():
                      print(f"[GitHub索引] 正在删除可能损坏的缓存目录: {self.cache_dir}")
                      try:
                          shutil.rmtree(self.cache_dir)
                      except Exception as rm_err:
                          print(f"[GitHub索引] 删除缓存目录失败: {rm_err}")

        # 如果缓存未启用或加载失败，则创建新索引
        if not loaded_from_cache:
            if not use_cache:
                print(f"[GitHub索引] 缓存已禁用，将重新创建索引")
            else:
                 # 检查缓存目录是否存在，以判断失败原因
                 if not self.cache_dir.exists():
                      print(f"[GitHub索引] 未找到有效缓存或缓存已被删除，将创建新索引")
                 else: # 缓存存在但加载失败（如模型不匹配）
                      print(f"[GitHub索引] 缓存加载失败（可能因不兼容或其他错误），将重新创建索引")


            # 加载文档
            print("[GitHub索引] 开始加载文档...")
            self.documents = self._load_documents_sync()

            if self.documents:
                print("[GitHub索引] 文档加载完成，开始创建索引...")
                try:
                    # 创建索引
                    self.index = self._create_index_sync(self.documents)
                    if self.index:
                        print("[GitHub索引] 索引创建成功。")
                        # 保存到缓存 (如果启用)
                        if use_cache:
                            print("[GitHub索引] 开始保存索引到缓存...")
                            self._save_to_cache()
                        # 创建 Query Engine
                        print("[GitHub索引] 开始创建查询引擎...")
                        # *** 关键改动 2: 创建 Query Engine 时确保 Settings 正确 ***
                        self.query_engine = self.index.as_query_engine()
                        if self.query_engine:
                             print("[GitHub索引] 查询引擎已成功创建。")
                        else:
                             print("[GitHub索引] 警告: index.as_query_engine() 返回了 None。")
                    else:
                        print("[GitHub索引] 警告: 索引创建函数返回了 None。无法创建查询引擎。")
                        self.query_engine = None # 明确标记
                except Exception as e:
                    print(f"[GitHub索引] 创建索引或查询引擎时出错: {e}")
                    traceback.print_exc()
                    self.index = None # 标记索引创建失败
                    self.query_engine = None
            else:
                 print("[GitHub索引] 警告: 未能加载任何文档，无法创建索引。")
                 self.index = None
                 self.query_engine = None

        # 最后检查 query_engine 是否有效
        if not self.query_engine:
             print(f"[GitHub索引] 警告: QA 系统 {self.cache_key} 初始化完成，但查询引擎不可用。")
        else:
             print(f"[GitHub索引] QA 系统 {self.cache_key} 初始化成功。")


    def _initialize_embedding_model(self):
        """初始化嵌入模型实例并存储在 self.embedding_model"""
        model_name = None
        if self.mode == "local":
            # 使用与创建索引时一致的模型
            model_name = "all-MiniLM-L6-v2" # 常见的 sentence-transformers 模型
            print(f"[GitHub索引] 准备初始化本地嵌入模型: {model_name}")
            try:
                # 可以在这里设置设备，例如 'cuda' 或 'mps' (for Apple Silicon)
                # device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
                # print(f"[GitHub索引] 使用设备: {device}")
                self.embedding_model = HuggingFaceEmbedding(
                    model_name=model_name,
                    # device=device # 传递设备参数
                )
                # 测试一下模型是否能加载和使用
                print("[GitHub索引] 测试嵌入模型加载...")
                _ = self.embedding_model.get_text_embedding("test")
                print(f"[GitHub索引] 本地嵌入模型 {model_name} 初始化成功。")
            except Exception as e:
                print(f"[GitHub索引] 初始化 HuggingFaceEmbedding ({model_name}) 失败: {e}")
                traceback.print_exc()
                self.embedding_model = None # 标记失败
        # 可以添加其他 mode 的处理逻辑
        # elif self.mode == "openai":
        #     try:
        #         from llama_index.embeddings.openai import OpenAIEmbedding
        #         print("[GitHub索引] 准备初始化 OpenAI 嵌入模型...")
        #         self.embedding_model = OpenAIEmbedding() # 假设 API Key/环境已配置
        #         _ = self.embedding_model.get_text_embedding("test")
        #         print("[GitHub索引] OpenAI 嵌入模型初始化成功。")
        #     except ImportError:
        #         print("[GitHub索引] 错误: 需要安装 llama-index-embeddings-openai 包才能使用 OpenAI 模型。")
        #         self.embedding_model = None
        #     except Exception as e:
        #         print(f"[GitHub索引] 初始化 OpenAIEmbedding 失败: {e}")
        #         traceback.print_exc()
        #         self.embedding_model = None
        else:
            print(f"[GitHub索引] 警告: 不支持的 mode '{self.mode}'，无法初始化嵌入模型。")
            self.embedding_model = None

    def _load_documents_sync(self):
        """同步加载文档"""
        try:
            github_token = os.environ.get("GITHUB_TOKEN")
            if not github_token:
                print("[GitHub索引] 错误: 无法获取GITHUB_TOKEN环境变量。请确保.env文件已正确配置。")
                return []

            print(f"[GitHub索引] 正在初始化 GitHub 客户端...")
            # 增加超时和重试可能有助于处理网络问题
            github_client = GithubClient(github_token=github_token, verbose=False) #, retries=3, timeout=60)

            # 配置过滤器
            filter_list = []
            if self.docs_folder_path: # 只有当路径非空时才添加目录过滤器
                 filter_list.append(
                     (
                         [self.docs_folder_path],
                         GithubRepositoryReader.FilterType.INCLUDE,
                     )
                 )
            filter_list.append(
                 (
                    [".md"], # 只包含 markdown 文件
                    GithubRepositoryReader.FilterType.INCLUDE,
                 )
            )

            self.reader = GithubRepositoryReader(
                github_client=github_client,
                owner=self.owner,
                repo=self.repo,
                use_parser=True, # 使用内置解析器
                verbose=False, #减少日志量
                filter_directories=filter_list[0] if self.docs_folder_path else None,
                filter_file_extensions=filter_list[1] if self.docs_folder_path else filter_list[0], # 根据是否有目录过滤器调整索引
                # concurrent_requests=5 # 可以调整并发数尝试加速，但可能增加 API rate limit 风险
            )

            folder_display = f"'{self.docs_folder_path}'" if self.docs_folder_path else "根目录"
            print(f"[GitHub索引] 开始从 {self.owner}/{self.repo}/{folder_display} (分支: {self.branch}) 加载文档...")
            print(f"[GitHub索引] 正在克隆或更新仓库 (这可能需要一些时间)...")
            start_time = time.time()
            # 捕获加载数据的特定错误
            try:
                documents = self.reader.load_data(branch=self.branch)
            except Exception as load_err:
                 print(f"[GitHub索引] reader.load_data 失败: {load_err}")
                 traceback.print_exc()
                 return []

            end_time = time.time()
            print(f"[GitHub索引] 文档加载耗时: {end_time - start_time:.2f} 秒")

            if not documents:
                print(f"[GitHub索引] 警告: 在 {self.owner}/{self.repo}/{folder_display} (分支: {self.branch}) 中未找到 .md 文档，或无权访问。")
            else:
                print(f"[GitHub索引] 成功加载了 {len(documents)} 个文档")
                extensions = {}
                for doc in documents:
                    # 提取文件名或路径以供参考
                    file_id = doc.doc_id if hasattr(doc, 'doc_id') else '未知文件'
                    if hasattr(doc, 'metadata') and 'file_path' in doc.metadata:
                        file_path = doc.metadata['file_path']
                        file_id = file_path # 用路径覆盖默认id
                        ext = os.path.splitext(file_path)[1]
                        extensions[ext] = extensions.get(ext, 0) + 1
                    # print(f"  - 加载: {file_id}") # 打印加载的文件名（可选）
                if extensions:
                    ext_info = ", ".join([f"{ext}: {count}" for ext, count in extensions.items()])
                    print(f"[GitHub索引] 文档类型统计: {ext_info}")
            return documents
        except Exception as e:
            print(f"[GitHub索引] 加载文档过程中发生意外错误: {e}")
            traceback.print_exc()
            return []


    def _generate_cache_key(self):
        """生成基于仓库信息的缓存键"""
        # 包含 mode 可能是个好主意，如果不同 mode 用不同模型
        # key_string = f"{self.owner}_{self.repo}_{self.branch}_{self.docs_folder_path}_{self.mode}"
        # 但当前设计是 mode 决定模型，模型信息已包含在元数据中用于检查，所以不一定需要
        key_string = f"{self.owner}_{self.repo}_{self.branch}_{self.docs_folder_path or 'root'}" # 用 'root' 表示根目录
        hash_obj = hashlib.md5(key_string.encode())
        # 使用更短的哈希，并确保 owner/repo 不包含非法字符
        safe_owner = "".join(c if c.isalnum() else "_" for c in self.owner)
        safe_repo = "".join(c if c.isalnum() else "_" for c in self.repo)
        return f"{safe_owner}_{safe_repo}_{hash_obj.hexdigest()[:8]}"

    def _save_to_cache(self):
        """将索引和元数据保存到本地缓存"""
        try:
            if self.index and self.embedding_model: # 确保索引和模型都存在
                self.cache_dir.mkdir(parents=True, exist_ok=True)
                print(f"[GitHub索引] 准备将索引保存到: {self.cache_dir}")

                # 获取嵌入模型信息
                model_name = getattr(self.embedding_model, 'model_name', '未知模型') # 更安全地获取名称
                actual_dim = 0
                try:
                    # 再次尝试获取维度
                    test_embedding = self.embedding_model.get_text_embedding("test_dimension")
                    actual_dim = len(test_embedding)
                    print(f"[GitHub索引] 缓存前测量嵌入模型维度: {actual_dim}")
                except Exception as e:
                    print(f"[GitHub索引] 测量嵌入维度失败: {e}. 尝试使用已知维度。")
                    # 备用方案：对于已知模型硬编码维度
                    if isinstance(self.embedding_model, HuggingFaceEmbedding) and "MiniLM-L6-v2" in model_name:
                        actual_dim = 384
                        print(f"[GitHub索引] 使用已知维度 {actual_dim} for {model_name}")
                    # elif isinstance(self.embedding_model, OpenAIEmbedding): # 示例
                    #     actual_dim = 1536 # 假设是 text-embedding-ada-002
                    #     print(f"[GitHub索引] 使用已知维度 {actual_dim} for OpenAI")
                    else:
                         print("[GitHub索引] 警告: 无法确定嵌入维度，缓存元数据可能不准确。")
                         actual_dim = 0 # 标记为未知

                # 保存索引
                print("[GitHub索引] 开始持久化存储上下文...")
                persist_start_time = time.time()
                self.index.storage_context.persist(persist_dir=str(self.cache_dir))
                persist_end_time = time.time()
                print(f"[GitHub索引] 存储上下文持久化完成，耗时: {persist_end_time - persist_start_time:.2f} 秒")


                embedding_model_info = {
                    "name": model_name,
                    "dimensions": actual_dim,
                    "class": type(self.embedding_model).__name__ # 保存模型类名，增加兼容性检查
                }

                metadata = {
                    "owner": self.owner,
                    "repo": self.repo,
                    "branch": self.branch,
                    "docs_folder_path": self.docs_folder_path,
                    "mode": self.mode, # 保存 mode
                    "timestamp": datetime.now().isoformat(), # 使用 datetime
                    "embedding_model": embedding_model_info,
                    "llama_index_version": getattr(Settings, '__version__', '未知版本') # 记录 LlamaIndex 版本
                }

                metadata_path = self.cache_dir / "metadata.json"
                print(f"[GitHub索引] 正在写入元数据文件: {metadata_path}")
                with open(metadata_path, "w", encoding="utf-8") as f:
                    json.dump(metadata, f, ensure_ascii=False, indent=2)

                print(f"[GitHub索引] 已成功缓存索引和元数据: {self.cache_key}")
                print(f"[GitHub索引] 使用嵌入模型: {model_name} ({embedding_model_info['class']}), 维度: {actual_dim if actual_dim > 0 else '未知'}")
                return True
            elif not self.embedding_model:
                 print("[GitHub索引] 错误: 无法保存缓存，因为嵌入模型未初始化。")
                 return False
            else: # index is None
                 print("[GitHub索引] 索引不存在，无法保存到缓存。")
                 return False
        except Exception as e:
            print(f"[GitHub索引] 缓存索引时出错: {e}")
            traceback.print_exc()
            return False

    def _load_from_cache(self):
        """从本地缓存加载索引，并确保使用正确的嵌入模型"""
        try:
            metadata_path = self.cache_dir / "metadata.json"
            vector_store_path = self.cache_dir / "default__vector_store.json" # 检查核心文件是否存在

            if not self.cache_dir.exists() or not metadata_path.exists() or not vector_store_path.exists():
                # print(f"[GitHub索引] 没有找到完整的缓存: {self.cache_key}") # 在 __init__ 中已有日志
                return False

            print(f"[GitHub索引] 找到缓存目录和元数据: {self.cache_dir}")
            with open(metadata_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)

            print(f"[GitHub索引] 索引时间: {metadata.get('timestamp', '未知')}")
            print(f"[GitHub索引] 缓存创建时 LlamaIndex 版本: {metadata.get('llama_index_version', '未知')}")

            cached_model_info = metadata.get('embedding_model', {})
            cached_model_name = cached_model_info.get('name', None)
            cached_dim = cached_model_info.get('dimensions', 0)
            cached_model_class = cached_model_info.get('class', None)

            # --- 兼容性检查 ---
            if not cached_model_name or not cached_model_class:
                print("[GitHub索引] 缓存元数据不完整 (缺少模型名称或类名)，无法安全加载。将重新创建索引。")
                shutil.rmtree(self.cache_dir) # 删除不完整的缓存
                return False

            # 检查当前初始化的模型是否与缓存匹配
            current_model_name = getattr(self.embedding_model, 'model_name', '未知模型')
            current_model_class = type(self.embedding_model).__name__
            current_dim = 0
            try:
                 test_emb = self.embedding_model.get_text_embedding("test_load")
                 current_dim = len(test_emb)
            except Exception as e:
                 print(f"[GitHub索引] 警告: 无法获取当前模型维度: {e}")
                 # 尝试使用已知维度
                 if isinstance(self.embedding_model, HuggingFaceEmbedding) and "MiniLM-L6-v2" in current_model_name:
                      current_dim = 384

            print(f"[GitHub索引] 缓存使用的嵌入模型: {cached_model_name} ({cached_model_class}), 维度: {cached_dim if cached_dim > 0 else '未知'}")
            print(f"[GitHub索引] 当前初始化的嵌入模型: {current_model_name} ({current_model_class}), 维度: {current_dim if current_dim > 0 else '未知'}")

            # 1. 检查模型类是否匹配
            if cached_model_class != current_model_class:
                 print(f"[GitHub索引] 缓存模型类 ({cached_model_class}) 与当前 ({current_model_class}) 不匹配。")
                 print(f"[GitHub索引] 删除不兼容的缓存并重新创建索引。")
                 shutil.rmtree(self.cache_dir)
                 return False

            # 2. 检查模型名称是否匹配 (对于 HuggingFace 很重要)
            if isinstance(self.embedding_model, HuggingFaceEmbedding) and cached_model_name != current_model_name:
                 print(f"[GitHub索引] 缓存模型名称 ({cached_model_name}) 与当前模型 ({current_model_name}) 不匹配。")
                 print(f"[GitHub索引] 删除不兼容的缓存并重新创建索引。")
                 shutil.rmtree(self.cache_dir)
                 return False

            # 3. 检查维度 (只有在两个维度都已知且 > 0 时才比较)
            if cached_dim > 0 and current_dim > 0 and cached_dim != current_dim:
                 print(f"[GitHub索引] 缓存的嵌入模型维度 ({cached_dim}) 与当前模型维度 ({current_dim}) 不匹配。")
                 print(f"[GitHub索引] 删除不兼容的缓存并重新创建索引。")
                 shutil.rmtree(self.cache_dir)
                 return False
            elif cached_dim == 0 and current_dim > 0:
                 print(f"[GitHub索引] 警告: 缓存元数据未记录维度，当前模型维度为 {current_dim}。将尝试加载，但可能存在风险。")
            elif current_dim == 0 and cached_dim > 0:
                 print(f"[GitHub索引] 警告: 无法获取当前模型维度，缓存维度为 {cached_dim}。将尝试加载，但可能存在风险。")

            # --- 加载索引 ---
            # *** 关键改动 3: 使用 ServiceContext 传递 embed_model ***
            print(f"[GitHub索引] 尝试从存储加载索引，并显式设置嵌入模型: {current_model_name}")
            try:
                # 确保使用在 __init__ 中初始化的 embedding_model 实例
                # 注意：如果缓存的 LlamaIndex 版本与当前版本差异过大，加载也可能失败
                self.embedding_model = HuggingFaceEmbedding(model_name=current_model_name)

                # 加载索引
                load_start_time = time.time()
                # 首先创建存储上下文
                storage_context = StorageContext.from_defaults(persist_dir=str(self.cache_dir))
                # 然后将其传递给load_index_from_storage
                self.index = load_index_from_storage(
                    storage_context
                    # 不再需要service_context参数
                )
                load_end_time = time.time()
                print(f"[GitHub索引] load_index_from_storage 完成，耗时: {load_end_time - load_start_time:.2f} 秒")


                if self.index:
                    print("[GitHub索引] 索引对象加载成功，开始创建查询引擎...")
                    # 创建查询引擎，它会自动使用 service_context 中的模型
                    self.query_engine = self.index.as_query_engine()
                    if self.query_engine:
                         # 标记文档为已加载（即使我们没有原始文档对象）
                         self.documents = ["cached"]
                         print(f"[GitHub索引] 查询引擎已成功从缓存索引创建。")
                         return True
                    else:
                         print("[GitHub索引] 错误: index.as_query_engine() 返回了 None。加载失败。")
                         # 清理已加载的 index 对象
                         self.index = None
                         return False
                else:
                    print("[GitHub索引] 错误: load_index_from_storage 返回了 None。加载失败。")
                    return False
            except Exception as e:
                print(f"[GitHub索引] 从存储加载索引失败: {e}")
                # 可能原因：版本不兼容、文件损坏、模型类变更等
                traceback.print_exc()
                print(f"[GitHub索引] 删除可能损坏或不兼容的缓存: {self.cache_dir}")
                try:
                     shutil.rmtree(self.cache_dir) # 删除损坏的缓存
                except Exception as rm_err:
                     print(f"[GitHub索引] 删除缓存目录失败: {rm_err}")
                return False

        except Exception as e:
            print(f"[GitHub索引] 从缓存加载索引时发生意外错误: {e}")
            traceback.print_exc()
            # 尝试删除缓存以避免循环错误
            if self.cache_dir.exists():
                 try:
                     shutil.rmtree(self.cache_dir)
                     print(f"[GitHub索引] 已删除可能导致错误的缓存目录: {self.cache_dir}")
                 except Exception as remove_err:
                     print(f"[GitHub索引] 删除缓存目录失败: {remove_err}")
            return False

    def _create_index_sync(self, documents):
        """同步创建索引"""
        try:
            if not documents:
                print("[GitHub索引] 警告: 没有提供文档来创建索引")
                return None

            print(f"[GitHub索引] 开始创建向量索引...共 {len(documents)} 个文档")
            start_total_time = time.time()

            # 文本分割器
            print("[GitHub索引] 阶段: 配置文本分割器")
            # chunk_size 和 chunk_overlap 可以根据文档特性和模型优化
            splitter = SentenceSplitter(chunk_size=1024, chunk_overlap=150) # 调整 chunk_size/overlap

            # 嵌入模型 (已由全局 Settings.embed_model 设置好)
            print(f"[GitHub索引] 阶段: 使用全局嵌入模型 {Settings.embed_model.model_name}")

            # 索引构建
            print("[GitHub索引] 阶段: 构建文档索引 (VectorStoreIndex.from_documents)")
            start_index_time = time.time()

            # 可以在这里指定嵌入批处理大小，覆盖模型默认值
            embed_batch_size = 32 # 根据你的硬件调整 (GPU显存/CPU核心数)
            print(f"[GitHub索引] 使用嵌入批处理大小: {embed_batch_size}")

            # 不再需要显式创建 ServiceContext，除非要覆盖 splitter 等
            # 全局 Settings.embed_model 会被自动使用

            index = VectorStoreIndex.from_documents(
                documents,
                transformations=[splitter], # 使用 transformations 参数传递分割器
                show_progress=True, # 显示 llama-index 的内置进度条
                embed_batch_size=embed_batch_size, # 指定批处理大小
                # use_async=True # 可以尝试异步嵌入，但可能与当前线程池冲突，需小心测试
            )

            end_index_time = time.time()
            print(f"[GitHub索引] 索引构建完成，耗时: {end_index_time - start_index_time:.2f} 秒")

            # 性能指标
            end_total_time = time.time()
            total_time = end_total_time - start_total_time
            docs_per_second = len(documents) / total_time if total_time > 0 else 0
            print(f"[GitHub索引] 索引创建总过程耗时: {total_time:.2f}秒")
            print(f"[GitHub索引] 平均处理速度: {docs_per_second:.2f} 文档/秒")

            print(f"[GitHub索引] 索引对象创建成功。")
            return index
        except Exception as e:
            print(f"[GitHub索引] 创建索引时出错: {e}")
            traceback.print_exc()
            return None


    def query_docs(self, query: str) -> str:
        """查询文档"""
        if self.query_engine is None:
            error_msg = "[GitHub索引] 抱歉，查询引擎不可用。"
            if self.index is None:
                 error_msg += " 索引未能成功加载或创建。"
            else:
                 error_msg += " 可能在初始化查询引擎时发生错误。"
            print(error_msg)
            return error_msg

        # *** 关键改动 4: 简化查询逻辑 ***
        # 理论上不需要再检查维度，因为 query_engine 创建时已保证一致性
        # 可以添加一个快速的 ping 测试 query_engine 是否工作

        try:
             print(f"[GitHub索引] 准备在线程池中执行查询: '{query}'")
             # 设置合理的超时时间，例如 60 秒
             timeout_seconds = 60
             future = _executor.submit(self._query_sync, query)
             response = future.result(timeout=timeout_seconds)
             print("[GitHub索引] 查询执行完成。")
             return response
        except concurrent.futures.TimeoutError:
            print(f"[GitHub索引] 查询超时（超过 {timeout_seconds} 秒）。")
            return f"抱歉，查询超时（>{timeout_seconds}s），请尝试简化查询或稍后再试。"
        except Exception as e:
            print(f"[GitHub索引] 查询文档时出错: {e}")
            traceback.print_exc()
             # 检查是否还是维度错误，给出提示
            if "shape" in str(e).lower() and "aligned" in str(e).lower():
                 # 提取维度信息（如果可能）
                 import re
                 match = re.search(r'\((\d+),?\).*\((\d+),?\)', str(e))
                 dims_info = f" ({match.group(1)} vs {match.group(2)})" if match else ""
                 print(f"[GitHub索引] 查询失败，仍然遇到维度不匹配错误{dims_info}！")
                 return f"查询失败，维度不匹配{dims_info}。\n这不应该发生，请尝试删除缓存目录 '{self.cache_dir}' 后重试。"

            # 对其他常见错误给出用户友好的提示
            elif "rate limit" in str(e).lower():
                 return "查询失败：达到了 API 调用频率限制。请稍后再试。"
            elif "authentication" in str(e).lower():
                 return "查询失败：身份验证错误。请检查您的 API 密钥或凭证。"

            return f"处理查询时发生错误: {str(e)}"

    def _query_sync(self, query: str) -> str:
        """执行同步查询（简化版）"""
        # 这个函数现在只负责调用 query_engine.query
        # 所有模型和配置应该在 query_engine 创建时已设定好
        try:
            print(f"[GitHub索引] (_query_sync) 调用 query_engine.query('{query}')...")
            response = self.query_engine.query(query)
            # 检查 response 类型，确保返回字符串
            if hasattr(response, 'response'):
                result = str(response.response)
            else:
                result = str(response)

            print(f"[GitHub索引] (_query_sync) 查询成功。") # 返回: {result[:100]}...")
            return result

        except Exception as e:
            # 在这里捕获并记录详细错误，然后重新抛出给上层处理
            print(f"[GitHub索引] (_query_sync) query_engine.query 执行失败: {str(e)}")
            # traceback.print_exc() # 上层会打印，这里可以不打
            raise e # 重新抛出，让 query_docs 捕获并处理

# --- 异步和同步入口函数 ---
async def ask_github_docs_async(
    query: str,
    owner: str,
    repo: str,
    branch: str,
    docs_folder_path: str = "", # 默认为空
    mode: str = "local",
    use_cache: bool = True
) -> str:
    """异步接口，在线程池中运行同步函数"""
    loop = asyncio.get_event_loop()
    # 使用 functools.partial 来正确传递参数给线程池中的函数
    func = partial(
        ask_github_docs, # 调用同步版本
        query=query,
        owner=owner,
        repo=repo,
        branch=branch,
        docs_folder_path=docs_folder_path,
        mode=mode,
        use_cache=use_cache,
    )
    # 增加错误处理
    try:
        return await loop.run_in_executor(_executor, func)
    except Exception as e:
        print(f"[GitHub索引] ask_github_docs_async 执行出错: {e}")
        traceback.print_exc()
        return f"异步查询处理错误: {str(e)}"


def ask_github_docs(
    query: str,
    owner: str, # 直接接收参数
    repo: str,
    branch: str,
    docs_folder_path: str = "", # 默认为空
    mode: str = "local",
    use_cache: bool = True,
) -> str:
    """
    同步接口，获取或创建 QA 系统并执行查询。
    """
    github_token = os.environ.get("GITHUB_TOKEN")
    if not github_token:
        print("[GitHub索引] 错误: GITHUB_TOKEN 未设置。")
        return "抱歉，GitHub令牌未设置。请确保环境变量GITHUB_TOKEN已正确设置。"

    print(f"\n[GitHub索引] ask_github_docs: 查询='{query}', 仓库={owner}/{repo}, 分支={branch}, 路径='{docs_folder_path}', 模式={mode}, 缓存={use_cache}")

    try:
        # 使用管理器获取或创建QA系统
        qa_system = github_qa_manager.get_or_create_qa_system(
            owner=owner,
            repo=repo,
            branch=branch,
            docs_folder_path=docs_folder_path,
            mode=mode,
            use_cache=use_cache
        )

        # 检查 QA 系统是否成功获取/创建
        if qa_system is None:
             # get_or_create_qa_system 内部已有日志
             print(f"[GitHub索引] ask_github_docs: 未能获取或创建 QA 系统 for {owner}/{repo}/{docs_folder_path}")
             return f"抱歉，无法为仓库 {owner}/{repo} (路径: '{docs_folder_path}') 初始化QA系统。"

        # 检查索引和查询引擎是否有效（双重检查）
        if qa_system.index is None or qa_system.query_engine is None:
            print(f"[GitHub索引] ask_github_docs: QA 系统 {owner}/{repo}/{docs_folder_path} 的索引或查询引擎无效。")
            return f"抱歉，QA系统 {owner}/{repo}/{docs_folder_path} 的索引或查询引擎无效。请检查初始化日志。"

        # 执行查询
        print(f"[GitHub索引] ask_github_docs: 获取到有效 QA 系统，开始查询...")
        return qa_system.query_docs(query)

    except Exception as e:
        print(f"[GitHub索引] ask_github_docs 执行过程中发生意外错误: {e}")
        traceback.print_exc()
        return f"处理查询时发生严重错误: {str(e)}"

def ask_github_docs_url(url, query):
    try:
        path_part = url.split("github.com/")[-1]
        parts = path_part.split("/")
        if len(parts) < 2:
            raise ValueError("URL 至少需要包含 owner/repo")
        owner = parts[0]
        repo = parts[1]
        branch = "main" # 默认分支
        docs_folder_path = "" # 默认根目录
        if len(parts) > 3 and parts[2] in ["tree", "blob"]:
            branch = parts[3]
            docs_folder_path = "/".join(parts[4:]).strip('/')
        elif len(parts) > 2:
            docs_folder_path = "/".join(parts[2:]).strip('/')
        return ask_github_docs(
            query=query,
            owner=owner,
            repo=repo,
            branch=branch,
            docs_folder_path=docs_folder_path,
            mode="local",
            use_cache=True
        )
    except Exception as e:
        print(f"[GitHub索引] ask_github_docs_url 执行过程中发生意外错误: {e}")
        traceback.print_exc()
        return f"处理查询时发生严重错误: {str(e)}"
# --- 主程序入口 (用于测试) ---
if __name__ == "__main__":
    # --- 配置测试参数 ---
    test_url = "https://github.com/jax-ml/jax/tree/main/docs"
    # test_url = "https://github.com/stepbystepcode/CodeWay/tree/main/docs" # 使用你的仓库
    test_query = "JAX 中的 jit 装饰器有什么用途？" # 你的测试问题
    test_mode = "local"
    use_cache_on_test = True # 设置为 True 测试缓存，False 测试无缓存创建

    print("="*50)
    print("   GitHub RAG 测试脚本启动")
    print("="*50)
    print(f"测试 URL: {test_url}")
    print(f"测试查询: {test_query}")
    print(f"测试模式: {test_mode}")
    print(f"使用缓存: {use_cache_on_test}")
    print("-"*50)

    # --- 从 URL 解析参数 ---
    try:
        # 移除协议头和域名，分割路径
        path_part = test_url.split("github.com/")[-1]
        parts = path_part.split("/")

        if len(parts) < 2:
            raise ValueError("URL 至少需要包含 owner/repo")

        owner = parts[0]
        repo = parts[1]

        # 查找 "tree" 或 "blob" 来确定分支和路径
        branch = "main" # 默认分支
        docs_folder_path = "" # 默认根目录

        if len(parts) > 3 and parts[2] in ["tree", "blob"]:
            branch = parts[3]
            docs_folder_path = "/".join(parts[4:]).strip('/')
        elif len(parts) > 2:
             # 如果没有 tree/blob，假定后面全是路径（这可能不准确）
             print(f"[警告] URL 格式不标准，无法明确识别分支。假定分支为 '{branch}'，路径为 '{'/'.join(parts[2:])}'")
             docs_folder_path = "/".join(parts[2:]).strip('/')


        print(f"解析参数:")
        print(f"  Owner: {owner}")
        print(f"  Repo: {repo}")
        print(f"  Branch: {branch}")
        print(f"  Docs Path: '{docs_folder_path}'")
        print("-"*50)

        # --- 执行测试 ---
        print(f"开始调用 ask_github_docs...")
        start_run_time = time.time()

        result = ask_github_docs(
            query=test_query,
            owner=owner,
            repo=repo,
            branch=branch,
            docs_folder_path=docs_folder_path,
            mode=test_mode,
            use_cache=use_cache_on_test
        )

        end_run_time = time.time()
        print("-"*50)
        print(f"查询结果:")
        print(result)
        print("-"*50)
        print(f"本次运行总耗时: {end_run_time - start_run_time:.2f} 秒")
        print("="*50)


    except ValueError as ve:
         print(f"URL 解析错误: {ve}")
         traceback.print_exc()
    except Exception as main_e:
         print(f"执行主程序时发生错误: {main_e}")
         traceback.print_exc()

    # --- 关闭线程池 ---
    # 在脚本结束时关闭，确保所有后台线程完成
    print("\n正在关闭线程池...")
    _executor.shutdown(wait=True)
    print("线程池已关闭。")
    print("="*50)
    print("   测试脚本结束")
    print("="*50)