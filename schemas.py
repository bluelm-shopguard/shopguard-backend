from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Union

# --- Model Endpoint Schemas ---

class ModelPermission(BaseModel):
    id: str = Field(default="modelperm-default")
    object: str = "model_permission"
    created: int
    allow_create_engine: bool = False
    allow_sampling: bool = True
    allow_logprobs: bool = True
    allow_search_indices: bool = False
    allow_view: bool = True
    allow_fine_tuning: bool = False
    organization: str = "*"
    group: Optional[str] = None
    is_blocking: bool = False

class ModelCard(BaseModel):
    id: str
    object: str = "model"
    created: int
    owned_by: str = "owner"
    permission: List[ModelPermission] = []
    root: Optional[str] = None
    parent: Optional[str] = None

class ModelList(BaseModel):
    object: str = "list"
    data: List[ModelCard] = []

# --- Chat Completions Schemas ---

class ChatMessage(BaseModel):
    role: str
    content: Union[str, List[Dict[str, Any]]] # 支持多模态内容

class UsageInfo(BaseModel):
    prompt_tokens: int = 0
    total_tokens: int = 0
    completion_tokens: Optional[int] = 0

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    top_p: Optional[float] = None
    stream: Optional[bool] = False
    user: Optional[str] = None
    extra: Optional[Dict[str, Any]] = None
    # 新增RAG控制参数
    enable_rag: Optional[bool] = True  # 默认开启RAG
    rag_top_k: Optional[int] = 1       # RAG检索数量

class ChatCompletionResponseChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: Optional[str] = "stop"

class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatCompletionResponseChoice]
    usage: UsageInfo