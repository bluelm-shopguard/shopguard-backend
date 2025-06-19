import logging
import time
import uuid
import json
import uvicorn
import base64
import os
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
from typing import Dict, Any

# 导入你的项目模块
from MultiModal import extract_text, interpret_image
from vivogpt import ask_vivogpt
from rag import VivoEmbeddingClient, KnowledgeBase, RAGSystem, ALL_KNOWLEDGE_EMBEDDING_DATA
from function_call import parse_function_call, call_web_search_api
from prompt import get_all_system_prompt, get_system_prompt

#加载环境变量
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
    
    if request.stream:
        raise HTTPException(status_code=400, detail="Stream mode is not supported yet")
    
    try:
        # 1. 提取用户ID和用户类型
        user_id = request.user or extract_user_id_from_messages([msg.dict() for msg in request.messages])
        user_type = determine_user_type([msg.dict() for msg in request.messages])
        
        logger.info(f"处理用户 {user_id} (类型: {user_type}) 的请求")
        
        # 2. 转换为与server.py兼容的消息格式
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

        # 3. 多模态输入处理 (完全复制server.py逻辑)
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
                                text_parts.append(f"[图片文字内容]:\n{ocr_text.strip()}")
                        
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
                                text_parts.append(f"[图片描述]:\n{img_desc.strip()}")
                
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

        logger.info(f"原始合并后的文本内容: {merged_text[:200]}...")

        # 5. RAG检索 (改为可选)
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

        # 6. 构造LLM输入 (完全复制原server.py逻辑)
        # 为历史记录存储原始用户消息
        original_user_message_for_history = {
            "role": "user",
            "content": merged_text
        }

        # 准备传递给大模型的内容，可能已用RAG上下文增强
        content_for_llm = merged_text
        if retrieved_rag_context:
            content_for_llm = f"请参考以下背景知识:\n---\n{retrieved_rag_context}\n---\n\n用户的原始问题是:\n{merged_text}"
            logger.info(f"传递给LLM的增强内容 (带RAG):\n{content_for_llm[:300]}...")
        else:
            logger.info(f"传递给LLM的内容 (无RAG):\n{content_for_llm[:300]}...")

        # 7. 会话历史管理
        if user_id not in conversation_history:
            conversation_history[user_id] = []

        max_history = 100
        history_messages = conversation_history[user_id][-(max_history * 2):]

        # 8. 准备extra参数
        extra_params = request.extra or {}
        extra_params.setdefault("temperature", request.temperature or 0.7)
        extra_params.setdefault("max_tokens", request.max_tokens or 1024)
        extra_params.setdefault("top_p", request.top_p or 1.0)

        # 9. 第一次LLM调用：判断是否需要工具
        all_system_prompt = get_all_system_prompt(user_type)
        current_llm_user_message = {
            "role": "user",
            "content": content_for_llm
        }
        
        messages_for_llm = [
            {"role": "system", "content": all_system_prompt},
            current_llm_user_message
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

        # 将用户的原始消息添加到历史记录
        if not history_messages or history_messages[-1].get("content") != original_user_message_for_history["content"]:
            conversation_history[user_id].append(original_user_message_for_history)

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

                # 提取搜索参数 (完全复制server.py逻辑)
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
            
            # 搜索结果摘要处理 (完全复制server.py逻辑)
            final_search_content_for_llm = ""
            try:
                core_result_str = json.dumps(core_result, ensure_ascii=False)
                
                if len(core_result_str) > 1500:
                    logger.info(f"搜索结果过长 ({len(core_result_str)} chars)，将进行摘要。")
                    
                    summarization_prompt = (
                        f"你是一个信息处理助手。请将以下联网搜索结果总结为一段保留核心信息的摘要，特别关注价格,品质,平台可信度等关键信息，以便后续用于回答用户关于'{search_query}'的问题。请直接输出摘要内容，不要添加任何额外解释。\n\n"
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

            # 11. 第二次LLM调用：生成最终回复 (完全复制server.py逻辑)
            system_prompt_for_final_answer = get_system_prompt(user_type)
            
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

            logger.info(f"messages_for_final_llm: {json.dumps(messages_for_final_llm, ensure_ascii=False, indent=2)}")

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
        else:
            # 普通回复 (没有 function call)
            final_reply_to_user = llm_response_raw
            
            conversation_history[user_id].append({
                "role": "assistant",
                "content": final_reply_to_user
            })

        # 12. 格式化为OpenAI响应
        response_message = ChatMessage(role="assistant", content=final_reply_to_user)
        choice = ChatCompletionResponseChoice(
            index=0, 
            message=response_message, 
            finish_reason="stop"
        )
        
        # Token计数估算
        prompt_tokens = sum(len(str(msg.get("content", ""))) for msg in messages_for_llm) // 4
        completion_tokens = len(final_reply_to_user) // 4
        usage = UsageInfo(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens
        )

        logger.info(f"请求处理完成，返回内容长度: {len(final_reply_to_user)}")
        logger.info(f"会话历史长度: {len(conversation_history[user_id])}")
        
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