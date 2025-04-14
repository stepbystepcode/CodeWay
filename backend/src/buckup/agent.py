import Agently
import os
from src.rag import ask_github_docs_url
# 创建一个Agent工厂实例
agent_factory = (
    Agently.AgentFactory()
        # 给Agent工厂实例提供设置项：
        ## 将默认模型请求客户端设置为OAIClient（我们为OpenAI兼容格式定制的请求客户端）
        .set_settings("current_model", "OAIClient")
        ## 提供你的模型API-KEY
        .set_settings("model.OAIClient.auth", { "api_key": "" })
        ## 指定你的模型Base-URL，如DeepSeek
        .set_settings("model.OAIClient.url", os.environ.get("LOCAL_API_BASE"))
        ## 指定你想要调用的具体模型
        .set_settings("model.OAIClient.options", { "model": "LLM-Research/gemma-3-27b-it" })
)

# 从Agent工厂实例中创建Agent实例
## 你可以创建很多个Agent实例，它们都会继承Agent工厂的设置项
## 也就是说，你不需要再为每一个Agent实例指定模型、输入授权
agent = agent_factory.create_agent()
"""定义并注册自定义工具"""
# 自定义工具函数及依赖
from datetime import datetime
import pytz
def get_current_datetime(timezone):
    tz = pytz.timezone(timezone)
    return datetime.now().astimezone(tz)
# 自定义工具信息字典
tool_info = {
    "tool_name": "ask_github_docs",
    "desc": "Query documentation in a GitHub repository and return the answer.",
    "args": {
        "query": (
            "str",
            "[*Required] Query to ask about the documentation",
        ),
        "url": (
            "str",
            "[*Required] URL of the GitHub repository",
        ),
    },
    "func": ask_github_docs_url
}
# 向Agent实例注册自定义工具
agent.register_tool(
    tool_name = tool_info["tool_name"],
    desc = tool_info["desc"],
    args = tool_info["args"],
    func = tool_info["func"],
)
"""发起请求"""
# print(agent.input("在https://github.com/jax-ml/jax/tree/main/docs 中，JAX 中的 jit 装饰器有什么用途？").start())  
# print(agent.input("在https://github.com/stepbystepcode/CodeWay/tree/main/docs 中，a=?b=?").start())  
