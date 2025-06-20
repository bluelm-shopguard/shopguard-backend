# vivogpt.py
import uuid
import time
import requests
import os
from dotenv import load_dotenv
from auth_util import gen_sign_headers

# 加载环境变量
load_dotenv()

APP_ID = os.getenv("VIVO_APP_ID")
APP_KEY = os.getenv("VIVO_APP_KEY")
URI = os.getenv("VIVOGPT_API_URI")  
STREAM_URI = os.getenv("VIVOGPT_API_STREAM_URI") 
DOMAIN = os.getenv("VIVOGPT_API_DOMAIN")  
METHOD = 'POST'

def ask_vivogpt(messages, extra, model='vivo-BlueLM-TB-Pro', session_id=None):
    """
    向大模型发起同步请求并返回 (content, time_cost)。
    出错时返回 (None, 错误信息)。
    """
    system_messages = [msg for msg in messages if msg.get("role") == "system"]
    filtered_messages = [msg for msg in messages if msg.get("role") != "system"]
    
    # 合并所有system消息内容作为systemPrompt
    system_prompt = "\n".join([msg.get("content", "") for msg in system_messages]) if system_messages else None
    
    # 确保每个消息都有contentType字段
    for msg in filtered_messages:
        if "contentType" not in msg:
            msg["contentType"] = "text"
            
    if not session_id:
        session_id = str(uuid.uuid4())

    request_id = str(uuid.uuid4())
    params = {'requestId': request_id}
    payload = {
        'messages': filtered_messages,
        'model': model,
        'sessionId': session_id,
        'extra': extra if extra is not None else {}
    }
    
    # 如果有system消息，添加systemPrompt字段
    if system_prompt:
        payload['systemPrompt'] = system_prompt

    headers = gen_sign_headers(APP_ID, APP_KEY, METHOD, URI, params)
    headers['Content-Type'] = 'application/json'
    url = f'https://{DOMAIN}{URI}'

    start_time = time.time()
    try:
        resp = requests.post(url, json=payload, headers=headers, params=params, timeout=100)
    except requests.RequestException as e:
        # 错误类型: RequestException (网络或请求构建问题)
        # 错误码: N/A (来自异常对象本身)
        return None, f'RequestException: {str(e)}'

    time_cost = time.time() - start_time

    # 尝试将响应体解析为JSON，因为API错误通常在JSON体中包含 'code' 和 'msg'
    response_body_text = resp.text # 存储响应文本，以备非JSON或JSON解析失败时使用
    """
    示例响应体:
    {
    "code": 0,
    "data": {
        "sessionId": "7b666a7aa0a811eeb5aad8bbc1c0d6bd",
        "requestId": "891483e6-3503-45db-808a-ab28672cc175",
        "content": "周海媚并没有去世，她依然活跃在演艺圈中。周海媚是中国香港影视女演员，出生于1966年，曾经在多部电视剧和电影中担任主演，如《倚天屠龙记》、《杨门女将之军令如山》等。",
        "provider": "vivo",
        "model": "vivo-BlueLM-TB-Pro"
    },
    "msg": "done."
    }
    """
    try:
        res_obj = resp.json()
    except ValueError: # 响应体不是有效的JSON
        res_obj = None

    if resp.status_code == 200:
        if res_obj is None:
            # 错误类型: API契约错误 (HTTP 200 但非JSON)
            # 错误码: HTTP 200 (但格式无效)
            return None, f'API Error: Received HTTP 200, but response body is not valid JSON. Response: {response_body_text}'

        api_code = res_obj.get('code')
        api_msg = res_obj.get('msg')

        if api_code == 0:
            # API指示成功 (code: 0)
            data_field = res_obj.get('data')
            if isinstance(data_field, dict) and 'content' in data_field:
                content = data_field.get('content', '') # 如果content为None或空，则默认为空字符串
                return content, time_cost
            else:
                # 错误类型: API契约错误 (HTTP 200, code 0, 但数据格式错误)
                # 错误码: API Code 0 (但数据问题)
                return None, f'API Success (Code: 0), but "data.content" is missing or "data" is not a valid dictionary. Response: {res_obj}'
        else:
            # API指示错误，尽管HTTP为200 (例如 code: 1007, 2001)
            # 错误类型: API逻辑错误
            # 错误码: api_code的值
            error_message = f'API Error - Code: {api_code}'
            if api_msg:
                error_message += f', Message: {api_msg}'
            else:
                error_message += f', Raw Response: {res_obj}' # 如果没有msg，提供原始响应对象
            return None, error_message
    else:
        # HTTP状态码指示错误 (例如 4xx, 5xx)
        # 错误类型: HTTP错误
        # 错误码: resp.status_code
        error_message = f'HTTP Error {resp.status_code}.'
        if res_obj: # 如果响应体是JSON
            api_code_from_json = res_obj.get('code') # 尝试从JSON中获取API特定的错误码
            api_msg_from_json = res_obj.get('msg')

            if api_code_from_json is not None: # 如果JSON中包含 'code'
                error_message += f' API Code: {api_code_from_json}'
            if api_msg_from_json:
                error_message += f', Message: {api_msg_from_json}'
            elif response_body_text: # 如果没有特定msg，但有原始文本
                 error_message += f', Details: {response_body_text}'
        elif response_body_text: # 响应体不是JSON，但有文本内容
            error_message += f' Details: {response_body_text}'
        # 如果res_obj为None且response_body_text为空，则只返回HTTP错误状态码信息
        return None, error_message
    
def ask_vivogpt_stream(messages, extra, model='vivo-BlueLM-TB-Pro', session_id=None):
    """
    向大模型发起流式请求并生成响应。
    """
    system_messages = [msg for msg in messages if msg.get("role") == "system"]
    filtered_messages = [msg for msg in messages if msg.get("role") != "system"]
    
    # 合并所有system消息内容作为systemPrompt
    system_prompt = "\n".join([msg.get("content", "") for msg in system_messages]) if system_messages else None
    
    # 确保每个消息都有contentType字段
    for msg in filtered_messages:
        if "contentType" not in msg:
            msg["contentType"] = "text"
            
    if not session_id:
        session_id = str(uuid.uuid4())

    request_id = str(uuid.uuid4())
    params = {'requestId': request_id}
    payload = {
        'messages': filtered_messages,
        'model': model,
        'sessionId': session_id,
        'extra': extra if extra is not None else {}
    }
    
    # 如果有system消息，添加systemPrompt字段
    if system_prompt:
        payload['systemPrompt'] = system_prompt

    stream_uri = STREAM_URI if STREAM_URI else URI  # 使用流式URI或默认URI
    
    # 使用流式URI
    headers = gen_sign_headers(APP_ID, APP_KEY, METHOD, stream_uri, params)
    headers['Content-Type'] = 'application/json'
    url = f'https://{DOMAIN}{stream_uri}'

    try:
        resp = requests.post(url, json=payload, headers=headers, params=params, stream=True, timeout=100)
        return resp
    except requests.RequestException as e:
        return None

