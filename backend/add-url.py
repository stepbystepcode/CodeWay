import requests
import os
import hashlib
import json
from pathlib import Path
from agno.agent import Agent
from agno.knowledge.document import DocumentKnowledgeBase
from agno.vectordb.lancedb import LanceDb
from agno.models.ollama import Ollama
from agno.embedder.ollama import OllamaEmbedder
from agno.document.base import Document
from agno.reranker.cohere import CohereReranker
import dotenv

dotenv.load_dotenv()

embedder = OllamaEmbedder(id="nomic-embed-text", dimensions=768)
vector_db = LanceDb(
    table_name="recipes",
    uri="db",
    embedder=embedder,
    reranker=CohereReranker(model="rerank-multilingual-v3.0"),
)

from getList import get_md_raw_urls

# 创建缓存目录
cache_dir = Path("./url_cache")
cache_dir.mkdir(exist_ok=True)

# 缓存索引文件路径
cache_index_path = cache_dir / "index.json"

# 加载缓存索引
if cache_index_path.exists():
    with open(cache_index_path, "r", encoding="utf-8") as f:
        cache_index = json.load(f)
else:
    cache_index = {}

def get_url_content(url):
    """获取URL内容，优先使用缓存"""
    # 生成URL的哈希值作为缓存文件名
    url_hash = hashlib.md5(url.encode()).hexdigest()
    cache_file = cache_dir / f"{url_hash}.txt"
    
    # 检查URL是否在缓存索引中
    if url in cache_index and cache_file.exists():
        print(f"Using cached content for: {url}")
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                return f.read()
        except Exception as e:
            print(f"Error reading cache for {url}: {e}")
    
    # 如果没有缓存或读取缓存失败，则下载内容
    print(f"Downloading content from: {url}")
    try:
        response = requests.get(url)
        content = response.text
        
        # 保存到缓存
        with open(cache_file, "w", encoding="utf-8") as f:
            f.write(content)
        
        # 更新缓存索引
        cache_index[url] = url_hash
        with open(cache_index_path, "w", encoding="utf-8") as f:
            json.dump(cache_index, f, indent=2)
        
        return content
    except Exception as e:
        print(f"Error downloading {url}: {e}")
        return f"Error: Could not download content from {url}. {str(e)}"

# 获取URLs列表
url_list = get_md_raw_urls("https://github.com/jax-ml/jax/tree/main/docs")

# 使用缓存功能获取文档内容
documents = [Document(content=get_url_content(url)) for url in url_list]
knowledge_base = DocumentKnowledgeBase(
    documents=documents,
    vector_db=vector_db,
    embedder=embedder,
    num_documents=1
)

knowledge_base.load(recreate=False)  # Comment out after first run

agent = Agent(model=Ollama(id="qwen2.5:32b"), knowledge=knowledge_base, show_tool_calls=True,markdown=True,debug_mode=True)
agent.print_response("According to the JAX API compatibility document, which legacy modules are explicitly listed that currently contain some private APIs without an underscore prefix, and whose modules and APIs are being actively deprecated, and whose deprecation process may not fully follow the standard 3-month deprecation period?", markdown=True)