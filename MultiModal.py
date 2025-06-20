# MultiModal.py
# 多模态图片理解与OCR文本提取工具
import uuid
import requests
import os
from auth_util import gen_sign_headers
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()


# vivo AI 平台分配的 APP_ID 和 APP_KEY
APP_ID = os.getenv('VIVO_APP_ID')
APP_KEY = os.getenv('VIVO_APP_KEY')

# 接口路径和域名
URI = os.getenv('MULTIMODAL_URI')
DOMAIN = os.getenv('MULTIMODAL_DOMAIN')
METHOD = 'POST'

def extract_text(image_base64, temperature=0.1, max_tokens=1024, timeout=15):
    """
    使用多模态大模型对图片进行OCR文字提取，仅返回原始文本内容。
    """
    # 处理 base64 字符串格式
    if not image_base64.startswith('data:image'):
        # 如果是纯base64，添加前缀
        clean_base64 = f"data:image/JPEG;base64,{image_base64.strip()}"
    else:
        # 如果已经有前缀，保持原样
        clean_base64 = image_base64.strip()
    
    prompt_text = "请提取图片中的所有文字内容，按原格式返回。忽略图片描述，只返回原始文本。"
    request_id = str(uuid.uuid4())
    params = {'requestId': request_id}
    payload = {
        'requestId': request_id,
        'sessionId': str(uuid.uuid4()),
        'model': 'vivo-BlueLM-V-2.0',
        'messages': [
            {
                "role": "user",
                "content": clean_base64,  # 使用完整的data:image格式
                "contentType": "image"
            },
            {
                "role": "user",
                "content": prompt_text,
                "contentType": "text"
            }
        ],
        'extra': {
            "temperature": temperature,
            "max_tokens": max_tokens
        }
    }
    headers = gen_sign_headers(APP_ID, APP_KEY, METHOD, URI, params)
    headers['Content-Type'] = 'application/json'
    url = f'https://{DOMAIN}{URI}'
    try:
        resp = requests.post(url, json=payload, headers=headers, params=params, timeout=timeout)
        if resp.status_code != 200:
            return None, f'HTTP error: {resp.status_code} - {resp.text}'
        res_obj = resp.json()
        if res_obj.get('code') != 0:
            return None, f'API error: {res_obj.get("msg")}'
        return res_obj['data'].get('content', ''), None
    except Exception as e:
        return None, f'Request exception: {str(e)}'

def interpret_image(
    image_base64,
    prompt_text=None,
    temperature=0.9,
    top_p=0.7,
    top_k=50,
    max_tokens=1024,
    repetition_penalty=1.02,
    stop=None,
    ignore_eos=False,
    skip_special_tokens=True,
    timeout=200
):
    """
    使用多模态大模型对图片进行内容理解，返回详细描述。
    """
    # 处理 base64 字符串格式
    if not image_base64.startswith('data:image'):
        # 如果是纯base64，添加前缀
        clean_base64 = f"data:image/JPEG;base64,{image_base64.strip()}"
    else:
        # 如果已经有前缀，保持原样
        clean_base64 = image_base64.strip()
    
    if prompt_text is None:
        prompt_text = (
            "不能超过800字.请描述这张图片的全部内容，着重关注和购物有关的内容,要求覆盖以下方面：\n"
            "1. 主要物体/人物：列出图片中出现的主要物体或人物，并简要说明其特征、动作、姿态、表情等；\n"
            "2. 场景和环境：描述图片的背景、地点、时间、氛围、色彩等环境信息；\n"
            "3. 关系与互动：如有多个元素，说明它们之间的关系或互动情况；\n"
            "4. 其他显著特征：如特殊标志、符号、颜色、光影效果等；\n"
            "5. 图片整体风格或用途：如是插画、照片、截图、广告等，请说明类型和可能用途。\n"
            "请按照上述结构分条详细描述，内容尽量全面、具体。\n"
        )
    if stop is None:
        stop = ["</end>"]

    request_id = str(uuid.uuid4())
    params = {'requestId': request_id}
    payload = {
        'requestId': request_id,
        'sessionId': str(uuid.uuid4()),
        'model': 'vivo-BlueLM-V-2.0',
        'messages': [
            {
                "role": "user",
                "content": clean_base64,  # 使用完整的data:image格式
                "contentType": "image"
            },
            {
                "role": "user",
                "content": prompt_text,
                "contentType": "text"
            }
        ],
        'extra': {
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "max_tokens": max_tokens,
            "repetition_penalty": repetition_penalty,
            "stop": stop,
            "ignore_eos": ignore_eos,
            "skip_special_tokens": skip_special_tokens
        }
    }
    headers = gen_sign_headers(APP_ID, APP_KEY, METHOD, URI, params)
    headers['Content-Type'] = 'application/json'
    url = f'https://{DOMAIN}{URI}'
    try:
        resp = requests.post(url, json=payload, headers=headers, params=params, timeout=timeout)
        if resp.status_code != 200:
            return None, f'HTTP error: {resp.status_code} - {resp.text}'
        res_obj = resp.json()
        if res_obj.get('code') != 0:
            return None, f'API error: {res_obj.get("msg")}'
        return res_obj['data'].get('content', ''), None
    except Exception as e:
        return None, f'Request exception: {str(e)}'