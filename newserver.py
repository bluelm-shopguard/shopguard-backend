import logging
import time
import uuid
import json
import uvicorn
import base64
import os
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse,StreamingResponse
from dotenv import load_dotenv
from typing import Dict, Any

# 导入项目模块
from MultiModal import extract_text, interpret_image
from vivogpt import ask_vivogpt,ask_vivogpt_stream
from rag import VivoEmbeddingClient, KnowledgeBase, RAGSystem, ALL_KNOWLEDGE_EMBEDDING_DATA
from function_call import parse_function_call, call_web_search_api
from prompt import get_shopping_function_call_prompt,get_normal_function_call_prompt ,get_system_prompt,shopping_relevance_prompt

# 加载环境变量
load_dotenv()

# 导入新的 schemas
from schemas import (
    ModelCard, ModelList, ChatMessage, ChatCompletionRequest,
    ChatCompletionResponse, ChatCompletionResponseChoice, UsageInfo
)

# --- 日志和应用初始化 ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="OpenAI-Compatible Server for vivo BlueLM",
    version="1.0.0",
)

def parse_sse_response(response):
    """解析vivo API的SSE流式响应"""
    full_content = ""
    
    try:
        for line in response.iter_lines():
            if line:
                line_str = line.decode('utf-8', errors='ignore').strip()
                logger.debug(f"收到流式数据行: {line_str}")
                
                # 处理data行
                if line_str.startswith('data:'):
                    data_content = line_str[5:].strip()  # 去掉'data:'前缀
                    
                    if data_content == '[DONE]':
                        logger.info("流式响应结束标记")
                        break
                    
                    if not data_content:  # 空行跳过
                        continue
                    
                    try:
                        data_json = json.loads(data_content)
                        
                        # 根据官方文档，优先处理message字段
                        message = data_json.get('message', '')
                        reply = data_json.get('reply', '')  # 干预时使用reply字段
                        
                        # 如果message为空但reply有内容（触发干预），使用reply
                        content_to_yield = message if message else reply
                        
                        if content_to_yield:
                            full_content += content_to_yield
                            yield content_to_yield
                            logger.debug(f"提取到消息片段: {content_to_yield}")
                        
                        # 检查是否有错误码
                        if 'code' in data_json and 'msg' in data_json:
                            error_msg = f"API错误 - Code: {data_json['code']}, Message: {data_json['msg']}"
                            logger.error(error_msg)
                            yield f"\n[{error_msg}]"
                            break
                            
                    except json.JSONDecodeError as e:
                        logger.warning(f"JSON解析失败: {e}, 原始数据: {data_content}")
                        continue
                
                # 处理event行
                elif line_str.startswith('event:'):
                    event_type = line_str[6:].strip()
                    logger.info(f"收到事件类型: {event_type}")
                    
                    if event_type == 'close':
                        logger.info("流式响应正常结束")
                        break
                    elif event_type == 'error':
                        logger.error("流式响应发生错误")
                        # 错误信息会在下一个data行中
                        continue
                    elif event_type == 'antispam':
                        logger.warning("触发内容干预")
                        # 干预信息会在下一个data行中
                        continue
    
    except Exception as e:
        logger.error(f"解析SSE响应时发生错误: {e}")
        yield f"\n[流式解析错误: {str(e)}]"

def generate_openai_stream(response, request_id, model, user_id, conversation_history):
    """生成OpenAI格式的流式响应 - 根据vivo API格式修复"""
    complete_content = ""
    chunk_count = 0
    
    try:
        # 检查响应状态
        if response.status_code != 200:
            error_msg = f"流式请求失败，状态码: {response.status_code}, 响应: {response.text}"
            logger.error(error_msg)
            
            # 发送错误响应
            error_data = {
                'id': request_id,
                'object': 'chat.completion.chunk',
                'created': int(time.time()),
                'model': model,
                'choices': [{
                    'index': 0,
                    'delta': {'content': f'[请求错误: {error_msg}]'},
                    'finish_reason': 'stop'
                }]
            }
            yield f"data: {json.dumps(error_data)}\n\n"
            yield "data: [DONE]\n\n"
            return
        
        # 开始流式响应 - 发送角色信息
        start_data = {
            'id': request_id,
            'object': 'chat.completion.chunk',
            'created': int(time.time()),
            'model': model,
            'choices': [{
                'index': 0,
                'delta': {'role': 'assistant'},
                'finish_reason': None
            }]
        }
        yield f"data: {json.dumps(start_data)}\n\n"
        
        # 解析并转发内容
        for chunk in parse_sse_response(response):
            if chunk:
                complete_content += chunk
                chunk_count += 1
                
                chunk_data = {
                    'id': request_id,
                    'object': 'chat.completion.chunk',
                    'created': int(time.time()),
                    'model': model,
                    'choices': [{
                        'index': 0,
                        'delta': {'content': chunk},
                        'finish_reason': None
                    }]
                }
                yield f"data: {json.dumps(chunk_data)}\n\n"
        
        logger.info(f"流式响应解析完成，共处理 {chunk_count} 个块，总内容长度: {len(complete_content)}")
        
        # 发送结束标记
        final_data = {
            'id': request_id,
            'object': 'chat.completion.chunk',
            'created': int(time.time()),
            'model': model,
            'choices': [{
                'index': 0,
                'delta': {},
                'finish_reason': 'stop'
            }]
        }
        yield f"data: {json.dumps(final_data)}\n\n"
        yield "data: [DONE]\n\n"
        
        # 将完整的回复添加到历史记录
        if complete_content.strip() and user_id:
            if user_id not in conversation_history:
                conversation_history[user_id] = []
            
            assistant_message = {
                "role": "assistant",
                "content": complete_content.strip()
            }
            conversation_history[user_id].append(assistant_message)
            
            # 限制历史记录长度
            max_history = 100
            if len(conversation_history[user_id]) > max_history * 2:
                conversation_history[user_id] = conversation_history[user_id][-(max_history * 2):]
            
            logger.info(f"流式响应完成，已将回复添加到用户 {user_id} 的历史记录")
            logger.info(f"- 完整内容长度: {len(complete_content)}")
            logger.info(f"- 当前用户历史记录条目数: {len(conversation_history[user_id])}")
            logger.info(f"- 完整内容预览: {complete_content[:200]}...")
        else:
            logger.warning(f"流式响应内容为空或用户ID无效。内容长度: {len(complete_content)}, 用户ID: {user_id}")
        
    except Exception as e:
        logger.error(f"生成流式响应时发生错误: {e}", exc_info=True)
        
        # 如果已经有部分内容，仍然保存到历史记录
        if complete_content.strip() and user_id:
            if user_id not in conversation_history:
                conversation_history[user_id] = []
            conversation_history[user_id].append({
                "role": "assistant",
                "content": complete_content.strip() + f"\n[流式输出中断: {str(e)}]"
            })
            logger.info(f"流式输出出错但已保存部分内容到历史记录: {len(complete_content)} 字符")
        
        # 发送错误信息
        error_data = {
            'id': request_id,
            'object': 'chat.completion.chunk',
            'created': int(time.time()),
            'model': model,
            'choices': [{
                'index': 0,
                'delta': {'content': f'\n[流式输出错误: {str(e)}]'},
                'finish_reason': 'stop'
            }]
        }
        yield f"data: {json.dumps(error_data)}\n\n"
        yield "data: [DONE]\n\n"
# --- 会话历史管理---
conversation_history: Dict[str, list] = {}

# --- RAG 系统初始化 ---
RAG_APP_ID = os.getenv('VIVO_APP_ID')
RAG_APP_KEY = os.getenv('VIVO_APP_KEY')
RAG_API_DOMAIN = os.getenv('RAG_API_DOMAIN')
RAG_API_URI = os.getenv('RAG_API_URI')

if not all([RAG_APP_ID, RAG_APP_KEY, RAG_API_DOMAIN, RAG_API_URI]):
    raise ValueError("请在.env文件中配置RAG_APP_ID,RAG_APP_KEY, RAG_API_DOMAIN 和 RAG_API_URI。")

embedding_client_rag = None
knowledge_base_rag = None
rag_system_instance = None

if True:#某些调用RAG的逻辑RAG_APP_ID != 'YOUR_VIVO_APP_ID' and RAG_APP_KEY != 'YOUR_VIVO_APP_KEY':
    try:
        embedding_client_rag = VivoEmbeddingClient(
            app_id=RAG_APP_ID,
            app_key=RAG_APP_KEY,
            domain=RAG_API_DOMAIN,
            uri=RAG_API_URI
        )
        
        knowledge_base_rag = KnowledgeBase()
        knowledge_base_rag.load_knowledge_from_list(ALL_KNOWLEDGE_EMBEDDING_DATA)
        
        if knowledge_base_rag.embeddings_matrix is not None and knowledge_base_rag.embeddings_matrix.shape[0] > 0:
            rag_system_instance = RAGSystem(embedding_client_rag, knowledge_base_rag)
            logger.info("RAG 系统初始化成功。")
        else:
            logger.warning("知识库为空或加载失败，RAG 系统将不可用。")
    except Exception as e:
        logger.error(f"RAG 系统初始化失败: {e}", exc_info=True)
        rag_system_instance = None
else:
    logger.warning("RAG_APP_ID 或 RAG_APP_KEY 未配置。RAG 系统将不可用。")

# --- 标准化错误处理 ---
@app.exception_handler(HTTPException)
async def handle_http_exception(request: Request, exc: HTTPException):
    """处理 HTTP 异常，返回 OpenAI 标准错误格式"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "message": exc.detail,
                "type": "invalid_request_error" if exc.status_code == 400 else "server_error",
                "code": exc.status_code
            }
        }
    )

@app.exception_handler(Exception)
async def handle_exception(request: Request, exc: Exception):
    """处理所有未捕获的异常"""
    logger.error(f"未处理的异常: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": {
                "message": "Internal server error",
                "type": "server_error",
                "code": 500
            }
        }
    )

# --- API 端点 ---

@app.get("/v1/models", response_model=ModelList)
async def list_models():
    """列出服务器可用的模型"""
    model_cards = [
        ModelCard(id="vivo-BlueLM-TB-Pro", created=int(time.time())),
        ModelCard(id="vivo-BlueLM-V-2.0", created=int(time.time())),
    ]
    return ModelList(data=model_cards)

def extract_user_id_from_messages(messages: list) -> str:
    """从消息中提取用户ID，如果没有则生成默认值"""
    return "default_user"

def determine_user_type(messages: list) -> str:
    """根据消息内容判断用户类型"""
    return "学生"

@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def create_chat_completion(request: ChatCompletionRequest):
    """处理聊天补全请求，完全复制原server.py的功能逻辑。"""
    request_id = f"chatcmpl-{uuid.uuid4()}"
    
    try:
        # 1. 提取用户ID和用户类型
        user_id = request.user or extract_user_id_from_messages([msg.model_dump() for msg in request.messages])
        user_type = determine_user_type([msg.model_dump() for msg in request.messages])
        
        logger.info(f"处理用户 {user_id} (类型: {user_type}) 的请求")
        
        converted_messages = []
        has_image = False
        
        for msg in request.messages:
            if msg.role == 'user':
                if isinstance(msg.content, list):
                    # OpenAI Vision 格式转换为 server.py 格式
                    for part in msg.content:
                        if isinstance(part, dict):
                            if part.get("type") == "text":
                                text_content = part.get("text", "")
                                if text_content:
                                    converted_messages.append({
                                        "contentType": "text",
                                        "content": text_content
                                    })
                            elif part.get("type") == "image_url":
                                url_data = part.get("image_url", {})
                                if isinstance(url_data, dict):
                                    url = url_data.get("url", "")
                                    if "base64," in url:
                                        image_base64 = url.split("base64,")[1]
                                        converted_messages.append({
                                            "contentType": "image",
                                            "content": image_base64
                                        })
                                        has_image = True
                                        logger.info("转换OpenAI Vision格式图片到server.py格式")
                elif isinstance(msg.content, str):
                    if msg.content.startswith("data:image") and "base64," in msg.content:
                        image_base64 = msg.content.split("base64,")[1]
                        converted_messages.append({
                            "contentType": "image", 
                            "content": image_base64
                        })
                        has_image = True
                        logger.info("转换base64图片字符串到server.py格式")
                    else:
                        converted_messages.append({
                            "contentType": "text",
                            "content": msg.content
                        })
            else:
                # 非用户消息保持原样，但转换格式
                converted_messages.append({
                    "contentType": "text",
                    "content": msg.content if isinstance(msg.content, str) else str(msg.content)
                })

        if not converted_messages:
            raise HTTPException(status_code=400, detail="No valid messages found in request")

        # 3. 多模态输入处理
        text_parts = []
        try:
            if has_image:
                # 处理图片消息 (先OCR，再图片理解)
                for msg in converted_messages:
                    if msg.get("contentType") == "image":
                        base64_str = msg.get("content", "")
                        if not base64_str:
                            continue
                        
                        logger.info("开始OCR文字提取...")
                        # OCR文字提取
                        ocr_text, ocr_error = extract_text(base64_str, temperature=0.1)
                        if ocr_error:
                            logger.error(f"OCR图片文字提取失败: {ocr_error}")
                        else:
                            logger.info(f"OCR提取成功，文字长度: {len(ocr_text) if ocr_text else 0}")
                            if ocr_text and ocr_text.strip():
                                text_parts.append(f"[用户发了一张图片,图片文字内容为]:\n{ocr_text.strip()}")
                        
                        logger.info("开始图片理解...")
                        # 图片理解 (使用默认prompt)
                        img_desc, img_error = interpret_image(
                            base64_str,
                            temperature=0.9
                        )
                        if img_error:
                            logger.error(f"图片理解失败: {img_error}")
                        else:
                            logger.info(f"图片理解成功，描述长度: {len(img_desc) if img_desc else 0}")
                            if img_desc and img_desc.strip():
                                text_parts.append(f"[用户发了张图片,图片描述为]:\n{img_desc.strip()}")
                
                # 文本消息放在图片内容之后
                for msg in converted_messages:
                    if msg.get("contentType") == "text":
                        content = msg.get("content", "").strip()
                        if content:
                            text_parts.append(content)
            else:
                # 纯文本消息
                for msg in converted_messages:
                    if msg.get("contentType") == "text":
                        content = msg.get("content", "").strip()
                        if content:
                            text_parts.append(content)
        except Exception as e:
            logger.exception("处理消息时发生错误")
            raise HTTPException(status_code=500, detail=f"处理消息时发生错误: {str(e)}")

        # 4. 合并文本内容
        merged_text = "\n".join(text_parts)
        if not merged_text.strip():
            raise HTTPException(status_code=400, detail="Request content is empty after processing")

        logger.info(f"原始合并后的文本内容: {merged_text[:]}...")

        shopping_check_messages = [
            {"role": "user", "content": shopping_relevance_prompt(merged_text)}
        ]

        logger.info("开始购物相关性判断...")

        shopping_relevance_response, relevance_error = ask_vivogpt(
            messages=shopping_check_messages,
            model=request.model,
            extra={"temperature": 0.1, "max_tokens": 10}
        )

        is_shopping_related = False
        if shopping_relevance_response:
            relevance_clean = shopping_relevance_response.strip().lower()
            is_shopping_related = "是" in relevance_clean or "yes" in relevance_clean
            logger.info(f"购物相关性判断结果: {shopping_relevance_response.strip()} -> {is_shopping_related}")
        else:
            logger.warning(f"购物相关性判断失败: {relevance_error}，默认为购物相关")
            is_shopping_related = True  # 默认为购物相关，避免误判
        
        # 会话历史管理
        if user_id not in conversation_history:
            conversation_history[user_id] = []

        # 确保会话历史不超过最大长度
        max_history = 100
        history_messages = conversation_history[user_id][-(max_history * 2):]

        # 为历史记录存储原始用户消息
        original_user_message_for_history = {
            "role": "user",
            "content": merged_text
        }

        # 将用户的原始消息添加到历史记录
        if not history_messages or history_messages[-1].get("content") != original_user_message_for_history["content"]:
            conversation_history[user_id].append(original_user_message_for_history)

        # 5. RAG检索 (改为可选)
        if is_shopping_related:
            retrieved_rag_context = ""
            
            # 检查是否启用RAG
            if request.enable_rag and rag_system_instance:
                try:
                    logger.info(f"RAG: 启用RAG检索，使用查询 \"{merged_text[:100]}...\" 进行检索")
                    rag_top_k = request.rag_top_k or 2
                    retrieved_rag_context = rag_system_instance.retrieve_and_format(merged_text, top_n=rag_top_k)
                    if retrieved_rag_context:
                        logger.info(f"RAG: 检索到的上下文长度: {len(retrieved_rag_context)}")
                        logger.info(f"RAG: 检索到的上下文:\n{retrieved_rag_context[:200]}...")
                    else:
                        logger.info("RAG: 未检索到相关上下文。")
                except Exception as e:
                    logger.error(f"RAG 检索过程中发生错误: {e}", exc_info=True)
                    retrieved_rag_context = ""
            elif not request.enable_rag:
                logger.info("RAG: 用户禁用了RAG检索功能")
            elif not rag_system_instance:
                logger.info("RAG: 系统未初始化或知识库为空，跳过 RAG 检索")
        
        else:
            retrieved_rag_context = ""
            logger.info("购物相关性判断结果为否，跳过 RAG 检索")

        # 6. 构造LLM输入

        # 准备传递给大模型的内容，可能已用RAG上下文增强
        content_for_llm = merged_text
        if retrieved_rag_context:
            content_for_llm = f"请参考以下背景知识:\n---\n{retrieved_rag_context}\n---\n\n用户的原始问题是:\n{merged_text}"
            logger.info(f"传递给LLM的增强内容 (带RAG):\n{content_for_llm[:300]}...")
        else:
            logger.info(f"传递给LLM的内容 (无RAG):\n{content_for_llm[:300]}...")

        # 8. 准备extra参数
        extra_params = request.extra or {}
        extra_params.setdefault("temperature", request.temperature or 0.7)
        extra_params.setdefault("max_tokens", request.max_tokens or 1024)
        extra_params.setdefault("top_p", request.top_p or 1.0)

        # 9. 第一次LLM调用：判断是否需要工具,以及和购物相关调用不同prompt
        if is_shopping_related:
            function_call_system_prompt = get_shopping_function_call_prompt(user_type)
        else:
            function_call_system_prompt = get_normal_function_call_prompt(user_type)

        messages_for_llm = [
            {"role": "user", 
             "content": f"这是用户的问题:{content_for_llm}"
            },
            {"role": "system",
            "content": function_call_system_prompt
            }
        ]

        logger.info("开始第一次LLM调用（工具判断）...")
        llm_response_raw, time_cost = ask_vivogpt(
            messages=messages_for_llm,
            model=request.model,
            extra=extra_params
        )

        if llm_response_raw is None:
            logger.error(f"function_call模型推理失败: {time_cost}")
            raise HTTPException(status_code=500, detail=f"function_call模型推理失败: {time_cost}")

        logger.info(f"function_call大模型推理成功: 耗时={time_cost:.2f}秒, 响应内容={llm_response_raw}")

        # 10. 判断是否需要函数调用
        func_call_str = parse_function_call(llm_response_raw)
        
        if func_call_str:
            logger.info("检测到函数调用，开始执行...")
            
            try:
                func_calls = json.loads(func_call_str)
                if isinstance(func_calls, list) and func_calls:
                    func_params = func_calls[0].get("parameters", {})
                elif isinstance(func_calls, dict):
                    func_params = func_calls.get("parameters", {})
                else:
                    func_params = {}
                    logger.warning(f"Function call string '{func_call_str}' 解析后不是预期的列表或字典结构。")

                # 提取搜索参数
                search_query = func_params.get("search_query", "测试搜索关键词")
                search_engine = func_params.get("search_engine", "search_std")
                search_intent = bool(func_params.get("search_intent", False))
                count = int(func_params.get("count", 10))
                search_domain_filter = func_params.get("search_domain_filter")
                search_recency_filter = func_params.get("search_recency_filter", "noLimit")
                content_size = func_params.get("content_size", "medium")
                request_id_param = func_params.get("request_id")

                function_result = call_web_search_api(
                    search_query=search_query,
                    search_engine=search_engine,
                    search_intent=search_intent,
                    count=count,
                    search_domain_filter=search_domain_filter,
                    search_recency_filter=search_recency_filter,
                    content_size=content_size,
                    request_id=request_id_param,
                    user_id=user_id
                )
            except json.JSONDecodeError as json_ex:
                logger.warning(f"Function call JSON解析失败: {json_ex}. Raw string: '{func_call_str}'")
                function_result = {"error": "invalid function call JSON format"}
            except Exception as ex:
                logger.warning(f"Function call参数提取或API调用失败: {ex}")
                function_result = {"error": f"function call processing error: {str(ex)}"}
            
            logger.info(f"web_search联网搜索返回: {json.dumps(function_result, ensure_ascii=False)}")
            
            core_result = function_result.get("search_result", function_result)
            
            # 搜索结果摘要处理
            final_search_content_for_llm = ""
            try:
                core_result_str = json.dumps(core_result, ensure_ascii=False)
                
                if len(core_result_str) > 1500:
                    logger.info(f"搜索结果过长 ({len(core_result_str)} chars)，将进行摘要。")
                    
                    if is_shopping_related:
                        summarization_prompt = (
                            f"你是一个信息处理助手。请将以下联网搜索结果总结为一段保留核心信息的摘要，特别关注价格,品质,平台可信度等和购物诈骗有关的的关键信息，以便后续用于回答用户关于'{search_query}'的问题。请直接输出摘要内容，不要添加任何额外解释。\n\n"
                            f"原始搜索结果:\n{core_result_str}"
                        )
                    
                    else:
                        summarization_prompt = (
                            f"你是一个信息处理助手。请将以下联网搜索结果总结为一段保留核心信息的摘要，以便后续用于回答用户关于'{search_query}'的问题。请直接输出摘要内容，不要添加任何额外解释。\n\n"
                            f"原始搜索结果:\n{core_result_str}"
                        )
                    
                    summarization_messages = [
                        {"role": "user", "content": summarization_prompt}
                    ]
                    
                    summary, summary_error = ask_vivogpt(
                        messages=summarization_messages,
                        model=request.model,
                        extra=extra_params
                    )
                    
                    if summary:
                        final_search_content_for_llm = summary
                        logger.info(f"搜索结果摘要成功: {final_search_content_for_llm[:200]}...")
                    else:
                        logger.warning(f"搜索结果摘要失败: {summary_error}。将使用原始搜索结果。")
                        final_search_content_for_llm = json.dumps({"search_result": core_result}, ensure_ascii=False)
                else:
                    logger.info("搜索结果长度适中，无需摘要，使用原始结果。")
                    final_search_content_for_llm = json.dumps({"search_result": core_result}, ensure_ascii=False)

            except Exception as e:
                logger.error(f"处理搜索结果摘要时发生意外错误: {e}", exc_info=True)
                final_search_content_for_llm = json.dumps({"search_result": core_result}, ensure_ascii=False)

            # 11. 第二次LLM调用：生成最终回复
            if is_shopping_related:
                system_prompt_for_final_answer = get_system_prompt(user_type)
            else:
                system_prompt_for_final_answer = "你是一个智能助手，旨在回答用户的问题。请根据用户的提问和提供的背景信息生成准确的回复。"

            messages_for_final_llm = [{"role": "system", "content": system_prompt_for_final_answer}]
            updated_history_messages = conversation_history[user_id][-(max_history * 2):-1]
            messages_for_final_llm.extend(updated_history_messages)
            messages_for_final_llm.append(original_user_message_for_history)
            messages_for_final_llm.append({
                "role": "assistant",
                "content": llm_response_raw
            })
            messages_for_final_llm.append({
                "role": "function",
                "name": "web_search",
                "content": final_search_content_for_llm
            })

            logger.info(f"最终给LLM的消息: {json.dumps(messages_for_final_llm, ensure_ascii=False, indent=2)}")

        else:
            # 普通回复 (没有 function call)
            if is_shopping_related:
                system_prompt_for_final_answer = get_system_prompt(user_type)
            else:
                system_prompt_for_final_answer = "你是一个智能助手，旨在回答用户的问题。"
            
            messages_for_final_llm = [{"role": "system", "content": system_prompt_for_final_answer}]
            updated_history_messages = conversation_history[user_id][-(max_history * 2):-1]
            messages_for_final_llm.extend(updated_history_messages)
            messages_for_final_llm.append(original_user_message_for_history)

        # ========== 这里决定是否使用流式输出 ==========
        if request.stream:
            # 流式响应
            logger.info("使用流式输出生成最终回复")
            
            stream_response = ask_vivogpt_stream(
                messages=messages_for_final_llm,
                model=request.model,
                extra=extra_params
            )
            
            if stream_response is None or stream_response.status_code != 200:
                raise HTTPException(status_code=500, detail="流式模型推理失败")

            return StreamingResponse(
                generate_openai_stream(stream_response, request_id, request.model, user_id, conversation_history),
                media_type="text/plain",
                headers={"Cache-Control": "no-cache", "Connection": "keep-alive"}
            )
        else:
            # 非流式响应（保持原有逻辑）
            final_answer_from_llm, error_message = ask_vivogpt(
                messages=messages_for_final_llm,
                model=request.model,
                extra=extra_params
            )
            
            if final_answer_from_llm is None:
                logger.error("最终模型推理失败")
                logger.error(f"错误信息: {error_message}")
                conversation_history[user_id].append({"role": "assistant", "content": "抱歉，我处理后续信息时遇到了点问题。"})
                raise HTTPException(status_code=500, detail="最终模型推理失败")
            
            logger.info(f"最终大模型推理成功: 响应内容={final_answer_from_llm}")
            final_reply_to_user = final_answer_from_llm

            conversation_history[user_id].append({
                "role": "assistant",
                "content": final_reply_to_user
            })
            
            # 返回标准响应
            response_message = ChatMessage(role="assistant", content=final_answer_from_llm)
            choice = ChatCompletionResponseChoice(
                index=0, 
                message=response_message, 
                finish_reason="stop"
            )
            
            prompt_tokens = sum(len(str(msg.get("content", ""))) for msg in messages_for_final_llm) // 4
            completion_tokens = len(final_answer_from_llm) // 4
            usage = UsageInfo(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens
            )
            
            return ChatCompletionResponse(
                id=request_id,
                created=int(time.time()),
                model=request.model,
                choices=[choice],
                usage=usage
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"处理聊天请求时发生错误: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Request processing failed: {str(e)}")

# --- 额外的兼容性端点 ---

@app.get("/")
async def root():
    """根路径，返回API信息"""
    return {
        "message": "OpenAI-Compatible Server for vivo BlueLM",
        "version": "1.0.0",
        "endpoints": ["/v1/models", "/v1/chat/completions"],
        "features": ["RAG", "MultiModal", "WebSearch", "ConversationHistory"]
    }

@app.get("/v1/health")
async def health_check():
    """健康检查端点"""
    return {
        "status": "healthy",
        "timestamp": int(time.time()),
        "rag_available": rag_system_instance is not None,
        "active_sessions": len(conversation_history),
        "system_info": {
            "rag_initialized": rag_system_instance is not None,
            "knowledge_base_size": len(ALL_KNOWLEDGE_EMBEDDING_DATA) if ALL_KNOWLEDGE_EMBEDDING_DATA else 0
        }
    }

@app.get("/v1/stats")
async def get_stats():
    """获取服务器统计信息"""
    return {
        "active_sessions": len(conversation_history),
        "total_messages": sum(len(history) for history in conversation_history.values()),
        "rag_status": "available" if rag_system_instance else "unavailable",
        "knowledge_base_entries": len(ALL_KNOWLEDGE_EMBEDDING_DATA) if ALL_KNOWLEDGE_EMBEDDING_DATA else 0
    }

# --- 运行服务器 ---
if __name__ == "__main__":
    logger.info("启动 OpenAI-Compatible FastAPI 服务器...")
    logger.info(f"RAG系统状态: {'可用' if rag_system_instance else '不可用'}")
    logger.info(f"知识库条目数: {len(ALL_KNOWLEDGE_EMBEDDING_DATA) if ALL_KNOWLEDGE_EMBEDDING_DATA else 0}")
    uvicorn.run(app, host="0.0.0.0", port=8000)