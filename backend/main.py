import os
import re
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr, Field
from supabase import create_client, Client, ClientOptions
from dotenv import load_dotenv
import httpx
from typing import Optional
from datetime import datetime
# from src.agent import agent
class GithubUrlRequest(BaseModel):
    url: str = Field(..., description="GitHub URL, e.g. https://github.com/jax-ml/jax/tree/main/docs")


# Load environment variables
load_dotenv()

app = FastAPI()
# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins in development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Supabase client
supabase_url = os.environ.get("SUPABASE_URL")
supabase_key = os.environ.get("SUPABASE_ANON_KEY")  # Changed from SUPABASE_KEY to SUPABASE_ANON_KEY

def get_supabase() -> Client:
    if not supabase_url or not supabase_key:
        raise HTTPException(status_code=500, detail="Supabase credentials not configured")
    return create_client(supabase_url, supabase_key,
        options=ClientOptions(
            postgrest_client_timeout=10,
            storage_client_timeout=10,
        )
    )
REQUEST_TIMEOUT: float = 30.0
custom_sync_transport = httpx.HTTPTransport(retries=0) # 可选：如果你不希望 httpx 自动重试
sync_timeout = httpx.Timeout(REQUEST_TIMEOUT, connect=30.0) # 总超时10秒，连接超时5秒
custom_sync_client = httpx.Client(timeout=sync_timeout, transport=custom_sync_transport)
# Authentication models
class UserLogin(BaseModel):
    email: EmailStr
    password: str

class UserRegister(BaseModel):
    email: EmailStr
    password: str

class VerifyRequest(BaseModel):
    email: EmailStr
    token: str
    username: str

class AuthResponse(BaseModel):
    token: str
    user_id: str

@app.get("/api/health")
def health_check():
    """Simple health check endpoint"""
    return {"status": "ok", "timestamp": datetime.now().isoformat()}


@app.post("/api/login", response_model=AuthResponse)
async def login(user_credentials: UserLogin, supabase: Client = Depends(get_supabase)):
    try:
        response = supabase.auth.sign_in_with_password({
            'email': user_credentials.email,
            'password': user_credentials.password,
        })
        
        return {
            "token": response.session.access_token,
            "user_id": response.user.id
        }
    except Exception as e:
        raise HTTPException(status_code=401, detail=f"Authentication failed: {str(e)}") # timed out


@app.post("/api/register", response_model=AuthResponse)
async def register(user_data: UserRegister, supabase: Client = Depends(get_supabase)):
    try:
        response = supabase.auth.sign_up({
            'email': user_data.email,
            'password': user_data.password,
        })
        
        if response.session is None:
            # Email confirmation required
            return {
                "token": "",  # No token until email confirmed
                "user_id": response.user.id
            }
        
        return {
            "token": response.session.access_token,
            "user_id": response.user.id
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Registration failed: {str(e)}")


@app.post("/api/verify", response_model=AuthResponse)
async def verify(verify_data: VerifyRequest, supabase: Client = Depends(get_supabase)):
    try:
        response = supabase.auth.verify_otp(
            {
                "email": verify_data.email,
                "token": verify_data.token,
                "type": "email",  # Hardcoded to email as per Supabase requirements
            }
        )
        
        if not response.session:
            raise HTTPException(status_code=400, detail="Verification failed: Invalid token")
            
        update_user = (
            supabase.table("profiles")
            .update({"username": verify_data.username, "email": verify_data.email})
            .eq("id", response.user.id)
            .execute()
        )
        print("updated_user", update_user)
        return {
            "token": response.session.access_token,
            "user_id": response.user.id
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Verification failed: {str(e)}")

@app.post("/api/add-url")
async def add_github_url(request: GithubUrlRequest):
    """Add a GitHub URL to create a vector database for its documentation"""
    try:
        # 解析GitHub URL以提取用户名、仓库名和文档路径
        url_pattern = r"https://github\.com/([^/]+)/([^/]+)(?:/tree/[^/]+)?(/[^?#]*)?" 
        match = re.search(url_pattern, request.url)
        
        if not match:
            raise HTTPException(status_code=400, detail="Invalid GitHub URL format")
        
        owner = match.group(1)
        repo = match.group(2)
        docs_path = match.group(3) or "/docs"
        
        # 如果路径以/开头，删除它
        if docs_path.startswith("/"):
            docs_path = docs_path[1:]
        
        # 如果路径为空，默认为docs
        if not docs_path:
            docs_path = "docs"
        
        print(f"Processing GitHub documentation: Owner={owner}, Repo={repo}, Path={docs_path}")
        
        # 使用GitHubQAManager工具管理QA系统实例
        from src.rag import github_qa_manager
        
        # 获取或创建QA系统并加载文档（这会触发向量索引的创建）
        qa_system = github_qa_manager.get_or_create_qa_system(
            owner=owner,
            repo=repo,
            docs_folder_path=docs_path,
            mode="local"
        )
        
        # 检查是否成功创建了索引
        if qa_system.index is None:
            raise HTTPException(status_code=500, detail="Failed to create vector index for the repository")
        
        return {
            "status": "success",
            "message": f"Successfully added documentation from {owner}/{repo}/{docs_path}",
            "details": {
                "owner": owner,
                "repo": repo,
                "docs_path": docs_path
            }
        }
    except Exception as e:
        import traceback
        print(f"Error processing GitHub URL: {str(e)}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error processing GitHub URL: {str(e)}")

# For development server (uvicorn/granian)
# Command: granian --interface asgi main:app

class ChatRequest(BaseModel):
    query: str = Field(..., description="Query to ask about the documentation")
    url: str = Field(..., description="URL of the GitHub repository")
    mode: str = Field(..., description="Mode to use for the query")

from src.github import ask_qa

@app.post("/api/ask-github-docs")
async def ask_github_docs(request: ChatRequest):
    response = await ask_qa(request.query, request.url, mode=request.mode)
    return {"response": response.content}