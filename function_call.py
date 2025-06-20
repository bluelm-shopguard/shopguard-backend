import requests
import os
import json
from dotenv import load_dotenv
from auth_util import gen_sign_headers

# 加载环境变量
load_dotenv()

def parse_function_call(answer: str):
    """
    解析 <APIs> ... </APIs> 结构，提取function call内容
    """
    if "<APIs>" in answer and "</APIs>" in answer:
        start_idx = answer.index("<APIs>") + len("<APIs>")
        end_idx = answer.index("</APIs>")
        return answer[start_idx:end_idx].strip()
    return None

def call_web_search_api(
    search_query,
    search_engine="search_std",
    search_intent=True,
    count=4,
    search_domain_filter=None,
    search_recency_filter="noLimit",
    content_size="small",
    request_id=None,
    user_id=None
):
    url = os.getenv("WEB_SEARCH_URL")
    payload = {
        "search_query": search_query,
        "search_engine": search_engine,
        "search_intent": search_intent,
    }
    if count is not None:
        payload["count"] = count
    if search_domain_filter:
        payload["search_domain_filter"] = search_domain_filter
    if search_recency_filter:
        payload["search_recency_filter"] = search_recency_filter
    if content_size:
        payload["content_size"] = content_size
    if request_id:
        payload["request_id"] = request_id
    if user_id:
        payload["user_id"] = user_id

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {os.getenv('WEB_SEARCH_API_KEY')}" # 注意：此Token可能需要更新或从配置中读取
    }

    try:
        resp = requests.post(url, data=json.dumps(payload), headers=headers, timeout=10)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        return {"error": f"Web Search 调用失败: {str(e)}"}
