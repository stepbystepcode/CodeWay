import requests
import hashlib
import json
import asyncio
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from agno.agent import Agent
from agno.knowledge.document import DocumentKnowledgeBase
from agno.vectordb.lancedb import LanceDb
from agno.models.ollama import Ollama
from agno.embedder.ollama import OllamaEmbedder
from agno.document.base import Document
from agno.reranker.cohere import CohereReranker
from agno.models.openai import OpenAIChat
import dotenv

dotenv.load_dotenv()

embedder = OllamaEmbedder(id="nomic-embed-text", dimensions=768)
vector_db = LanceDb(
    table_name="recipes",
    uri="db",
    embedder=embedder,
    reranker=CohereReranker(model="rerank-multilingual-v3.0"),
)

from src.get_list import get_md_raw_urls

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


async def async_get_url_content(url):
    """异步获取URL内容，优先使用缓存"""
    # 使用同步函数，但在线程池中运行
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, get_url_content, url)


def download_parallel_with_threading(url_list, max_workers=10):
    """使用ThreadPoolExecutor进行并行下载"""
    start_time = time.time()
    documents = []
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务
        future_to_url = {executor.submit(get_url_content, url): url for url in url_list}
        
        # 获取结果
        for future in as_completed(future_to_url):
            url = future_to_url[future]
            try:
                content = future.result()
                documents.append(Document(content=content))
                print(f"Downloaded {url} (ThreadPool)")
            except Exception as e:
                print(f"Error processing {url}: {e}")
    
    print(f"Total download time with ThreadPoolExecutor: {time.time() - start_time:.2f} seconds")
    return documents


async def download_parallel_with_taskgroup(url_list, max_concurrent=10):
    """使用Python 3.13的TaskGroup特性进行并行下载"""
    start_time = time.time()
    documents = []
    
    # 使用分组处理，避免同时创建太多任务
    for i in range(0, len(url_list), max_concurrent):
        batch = url_list[i:i+max_concurrent]
        
        async with asyncio.TaskGroup() as tg:
            # 为每个URL创建一个任务
            tasks = [tg.create_task(async_get_url_content(url)) for url in batch]
        
        # 处理完成的任务结果
        for task, url in zip(tasks, batch):
            try:
                content = task.result()
                documents.append(Document(content=content))
                print(f"Downloaded {url} (TaskGroup)")
            except Exception as e:
                print(f"Error processing {url}: {e}")
    
    print(f"Total download time with TaskGroup: {time.time() - start_time:.2f} seconds")
    return documents

# 获取URLs列表

# 选择并行下载方法
# 方法1: 使用Python 3.13的TaskGroup (异步方式)
async def ask_qa(message, url, mode='local'):
    url_list = get_md_raw_urls(url)
    documents = await download_parallel_with_taskgroup(url_list, max_concurrent=10)
    
    
    if mode == 'local':
        knowledge_base = DocumentKnowledgeBase(
            documents=documents,
            vector_db=vector_db,
            embedder=embedder,
            num_documents=1
        )
        knowledge_base.load(recreate=False)  # Comment out after first run
        print("Using Qwen model")
        agent = Agent(
            model=Ollama(id="qwen2.5:32b"),
            knowledge=knowledge_base,
            show_tool_calls=True,
            markdown=True,
            debug_mode=True
        )
    else:
        knowledge_base = DocumentKnowledgeBase(
            documents=documents,
            vector_db=vector_db,
            embedder=embedder,
            num_documents=3
        )
        knowledge_base.load(recreate=False)  # Comment out after first run
        print("Using OpenAI model")
        agent = Agent(
            model=OpenAIChat(id="gpt-4o"),
            knowledge=knowledge_base,
            show_tool_calls=True,
            markdown=True,
            debug_mode=True
        )
    return agent.run(message, markdown=True)

if __name__ == "__main__":
    asyncio.run(ask_qa("According to the JAX API compatibility document, which legacy modules are explicitly listed that currently contain some private APIs without an underscore prefix, and whose modules and APIs are being actively deprecated, and whose deprecation process may not fully follow the standard 3-month deprecation period?", "https://github.com/jax-ml/jax/tree/main/docs"))