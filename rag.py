# encoding: utf-8
import requests
import numpy as np
import json
import logging
import os # 新增导入 os
from auth_util import gen_sign_headers # 确保 auth_util.py 在同一目录或PYTHONPATH中

logger = logging.getLogger(__name__)

# --- 从 JSON 文件加载知识库数据 ---
ALL_KNOWLEDGE_EMBEDDING_DATA = []
DEFAULT_KNOWLEDGE_FILE = "knowledge_base_embeddings/all_knowledge_embeddings.json"

def load_knowledge_from_json(file_path: str) -> list:
    """从指定的 JSON 文件加载知识库数据。"""
    data = []
    # 尝试相对于当前脚本文件路径查找JSON文件
    current_dir = os.path.dirname(os.path.abspath(__file__))
    absolute_file_path = os.path.join(current_dir, file_path)

    if not os.path.exists(absolute_file_path):
        logger.error(f"知识库文件未找到: {absolute_file_path}")
        return data # 返回空列表

    try:
        with open(absolute_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if not isinstance(data, list):
            logger.error(f"知识库文件 {absolute_file_path} 的内容不是一个列表。")
            return []
        logger.info(f"成功从 {absolute_file_path} 加载 {len(data)} 条知识库条目。")
    except json.JSONDecodeError as e:
        logger.error(f"解析知识库文件 {absolute_file_path} 失败: {e}")
        return []
    except Exception as e:
        logger.error(f"加载知识库文件 {absolute_file_path} 时发生未知错误: {e}")
        return []
    return data

# 在模块加载时执行加载操作
ALL_KNOWLEDGE_EMBEDDING_DATA = load_knowledge_from_json(DEFAULT_KNOWLEDGE_FILE)

if not ALL_KNOWLEDGE_EMBEDDING_DATA:
    logger.warning(
        f"未能从 {DEFAULT_KNOWLEDGE_FILE} 加载任何知识库数据。"
        "RAG 系统可能无法正常工作，除非数据在其他地方被正确加载。"
        "请确保 'all_knowledge_embeddings.json' 文件存在于 server 目录下且格式正确。"
    )
# --- 知识库数据加载结束 ---


class VivoEmbeddingClient:
    def __init__(self, app_id, app_key, domain, uri, method='POST'):
        self.app_id = app_id
        self.app_key = app_key
        self.domain = domain
        self.uri = uri
        self.method = method
        self.url = f'https://{self.domain}{self.uri}'

    def get_embeddings(self, sentences: list):
        if not sentences:
            return []
    
        params = {}
        post_data = {
            "model_name": "m3e-base",
            "sentences": sentences
        }
        
        try:
            headers = gen_sign_headers(self.app_id, self.app_key, self.method, self.uri, params)
            headers['Content-Type'] = 'application/json'

            response = requests.post(self.url, json=post_data, headers=headers, timeout=20)
            response.raise_for_status()
            response_json = response.json()

            # 修改响应解析逻辑
            if "data" in response_json and isinstance(response_json["data"], list):
                # 直接处理 data 字段中的向量数据
                vectors = response_json["data"]
                if vectors and all(isinstance(vec, list) for vec in vectors):
                    return [np.array(emb) for emb in vectors]
            
            # 如果有 code 字段，按原逻辑处理
            if response_json.get("code") == 0:
                # 尝试从常见的响应结构中提取向量
                vectors = None
                if "result" in response_json and "vectors" in response_json["result"]:
                    vectors = response_json["result"]["vectors"]
                elif "data" in response_json and "embeddings" in response_json["data"]:
                    vectors = response_json["data"]["embeddings"]
                elif "embeddings" in response_json and isinstance(response_json["embeddings"], list):
                    vectors = response_json["embeddings"]
                elif "vectors" in response_json and isinstance(response_json["vectors"], list):
                    vectors = response_json["vectors"]
                else: # 尝试更通用的查找
                    def find_embeddings_list(data_node):
                        if isinstance(data_node, list) and data_node and all(isinstance(el, list) for el in data_node):
                            if all(isinstance(num, (float, int)) for el_list in data_node for num in el_list):
                                return data_node
                        if isinstance(data_node, dict):
                            for k, v_node in data_node.items():
                                if k in ["embeddings", "vectors", "embedding_vectors"] and isinstance(v_node, list):
                                    if v_node and all(isinstance(el, list) for el in v_node):
                                         if all(isinstance(num, (float, int)) for el_list in v_node for num in el_list):
                                            return v_node
                                res = find_embeddings_list(v_node)
                                if res: return res
                        return None
                    vectors = find_embeddings_list(response_json)

                if vectors is not None:
                    return [np.array(emb) for emb in vectors]
                
                logger.error(f"无法从API响应中提取向量。Code: {response_json.get('code')}, Msg: {response_json.get('message', response_json.get('msg', 'N/A'))}. Response: {json.dumps(response_json, ensure_ascii=False)}")
                return []
            else:
                logger.error(f"Embedding API 调用失败。Code: {response_json.get('code')}, Msg: {response_json.get('message', response_json.get('msg', 'N/A'))}. Response: {json.dumps(response_json, ensure_ascii=False)}")
                return []

        except requests.exceptions.RequestException as e:
            logger.error(f"调用 Embedding API 时发生网络错误: {e}")
            return []
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            logger.error(f"解析 Embedding API 响应时出错: {e}. Response text: {response.text if 'response' in locals() else 'N/A'}")
            return []


class KnowledgeBase:
    def __init__(self):
        self.knowledge_entries = []
        self.embeddings_matrix = None
        self.texts = []

    def load_knowledge_from_list(self, knowledge_data: list):
        if not knowledge_data:
            logger.warning("知识库数据列表为空。")
            self.knowledge_entries = []
            self.embeddings_matrix = None
            self.texts = []
            return

        valid_entries = []
        embeddings_list = []
        texts_list = []
        expected_dim = None

        for i, entry in enumerate(knowledge_data):
            if isinstance(entry, dict) and "text" in entry and "embedding" in entry:
                embedding_vector = entry["embedding"]
                if isinstance(embedding_vector, list) and all(isinstance(x, (int, float)) for x in embedding_vector):
                    if expected_dim is None:
                        expected_dim = len(embedding_vector)
                    elif len(embedding_vector) != expected_dim:
                        logger.warning(f"知识库条目 {i} 的向量维度 ({len(embedding_vector)}) 与预期 ({expected_dim}) 不符。已跳过。")
                        continue
                    
                    valid_entries.append(entry)
                    embeddings_list.append(np.array(embedding_vector, dtype=np.float32))
                    texts_list.append(str(entry["text"]))
                else:
                    logger.warning(f"知识库条目 {i} 的 embedding 格式无效或非数值类型。已跳过。内容: {embedding_vector}")
            else:
                logger.warning(f"知识库条目 {i} 不是字典或缺少 'text'/'embedding' 键。已跳过。条目内容: {entry}")
        
        self.knowledge_entries = valid_entries
        if embeddings_list:
            try:
                self.embeddings_matrix = np.array(embeddings_list)
                self.texts = texts_list
                logger.info(f"成功加载 {len(self.knowledge_entries)} 条知识库条目到 KnowledgeBase。向量维度: {expected_dim if expected_dim else 'N/A'}.")
            except Exception as e:
                logger.error(f"将 embeddings_list 转换为 NumPy 数组时出错: {e}")
                self.embeddings_matrix = None
                self.texts = []
        else:
            logger.warning("未找到有效的知识库条目进行加载到 KnowledgeBase。")
            self.embeddings_matrix = None
            self.texts = []


    def _cosine_similarity(self, query_vec: np.ndarray, doc_matrix: np.ndarray):
        if query_vec is None or doc_matrix is None or doc_matrix.shape[0] == 0 or query_vec.ndim != 1 or doc_matrix.ndim != 2 or query_vec.shape[0] != doc_matrix.shape[1]:
            logger.warning(f"余弦相似度计算的输入无效。Query_vec shape: {query_vec.shape if query_vec is not None else 'None'}, Doc_matrix shape: {doc_matrix.shape if doc_matrix is not None else 'None'}")
            return np.array([])
        
        query_norm = np.linalg.norm(query_vec)
        if query_norm == 0:
            return np.zeros(doc_matrix.shape[0])
        
        query_vec_normalized = query_vec / query_norm
        
        doc_norms = np.linalg.norm(doc_matrix, axis=1)
        doc_matrix_normalized = np.zeros_like(doc_matrix, dtype=float)
        
        valid_indices = doc_norms > 1e-8 # 使用一个小的阈值避免除以零
        doc_matrix_normalized[valid_indices] = doc_matrix[valid_indices] / doc_norms[valid_indices, np.newaxis]
        
        similarities = np.dot(doc_matrix_normalized, query_vec_normalized)
        return similarities

    def find_similar_texts(self, query_embedding: np.ndarray, top_n=3):
        if self.embeddings_matrix is None or self.embeddings_matrix.shape[0] == 0:
            return []
        if query_embedding is None:
            logger.warning("查询向量为 None。")
            return []

        similarities = self._cosine_similarity(query_embedding, self.embeddings_matrix)
        
        if similarities.size == 0:
            return []

        num_items_to_return = min(top_n, len(similarities))
        if num_items_to_return == 0:
            return []
            
        top_indices = np.argsort(similarities)[-num_items_to_return:][::-1]

        results = []
        for i in top_indices:
            if similarities[i] > 0: # 可以根据需要调整相似度阈值
                # 获取完整的知识库条目信息
                entry = self.knowledge_entries[i]
                results.append({
                    "text": entry.get("text", ""),
                    "riskType": entry.get("riskType", "未知风险"),
                    "similarity": float(similarities[i]),
                })
        return results

class RAGSystem:
    def __init__(self, embedding_client: VivoEmbeddingClient, knowledge_base: KnowledgeBase):
        self.embedding_client = embedding_client
        self.knowledge_base = knowledge_base
        if self.knowledge_base.embeddings_matrix is None or self.knowledge_base.embeddings_matrix.shape[0] == 0:
            logger.warning("RAGSystem 初始化：知识库为空。RAG检索将不可用。")


    def retrieve_and_format(self, query_text: str, top_n=3):
        if not query_text.strip():
            logger.warning("RAG: 查询文本为空。")
            return ""
        
        if self.knowledge_base.embeddings_matrix is None or self.knowledge_base.embeddings_matrix.shape[0] == 0:
            logger.info("RAG: 知识库为空，无法执行检索。")
            return ""

        query_embeddings = self.embedding_client.get_embeddings([query_text])

        if not query_embeddings:
            logger.warning(f"RAG: 无法获取查询 '{query_text[:50]}...' 的向量。")
            return ""
        
        query_embedding = query_embeddings[0]

        similar_docs_info = self.knowledge_base.find_similar_texts(query_embedding, top_n=top_n)

        if not similar_docs_info:
            return ""

        formatted_texts = []
        for i, doc_info in enumerate(similar_docs_info):
            risk_type = doc_info.get('riskType', '未知风险')
            text_content = doc_info.get('text', '')
            similarity = doc_info.get('similarity', 0.0)
            
            # 格式化为明确的知识库参考信息
            formatted_text = f"【{risk_type}】的知识库参考信息 (相似度: {similarity:.2f}):\n{text_content}"
            formatted_texts.append(formatted_text)
        
        return "\n\n".join(formatted_texts)
