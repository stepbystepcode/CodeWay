import requests
from urllib.parse import urlparse
import dotenv
import os

dotenv.load_dotenv()
GITHUB_TOKEN = os.getenv('GITHUB_TOKEN')

def extract_github_info(url: str):
    """
    从 GitHub 文件夹 URL 中提取 owner, repo, branch, path
    """
    parsed = urlparse(url)
    parts = parsed.path.strip('/').split('/')
    if 'tree' not in parts:
        raise ValueError("URL must include '/tree/' to specify branch and path")

    owner = parts[0]
    repo = parts[1]
    tree_index = parts.index('tree')
    branch = parts[tree_index + 1]
    path = '/'.join(parts[tree_index + 2:])
    return owner, repo, branch, path

def get_repo_tree(owner: str, repo: str, branch: str):
    """
    获取指定仓库某个分支的文件树（递归）
    """
    url = f'https://api.github.com/repos/{owner}/{repo}/git/trees/{branch}?recursive=1'
    resp = requests.get(url, headers={"Authorization": f"token {GITHUB_TOKEN}"})
    if resp.status_code == 200:
        return resp.json().get('tree', [])
    else:
        raise Exception(f"Failed to get repo tree: {resp.status_code} - {resp.text}")

def get_md_raw_urls(github_folder_url: str):
    """
    主函数：输入 GitHub 文件夹链接，输出该目录下所有 .md 文件的 raw URL
    """
    owner, repo, branch, path_prefix = extract_github_info(github_folder_url)
    tree = get_repo_tree(owner, repo, branch)

    # 构造 raw 基础路径前缀
    raw_base_url = f'https://raw.githubusercontent.com/{owner}/{repo}/{branch}/'

    md_urls = []
    for item in tree:
        if item['type'] == 'blob' and item['path'].endswith('.md'):
            # 路径在指定文件夹下
            if item['path'].startswith(path_prefix):
                md_urls.append(raw_base_url + item['path'])

    return md_urls

# 示例
if __name__ == "__main__":
    github_url = "https://github.com/jax-ml/jax/tree/main/docs"
    md_raw_urls = get_md_raw_urls(github_url)
    print(md_raw_urls)