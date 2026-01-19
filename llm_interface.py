"""
LLM接口实现 - 最终修复版（匹配讯飞4.0Ultra的domain参数）
解决10004错误：'$.parameter.chat.domain' value is invalid
"""

import json
import os
import ssl
import time
import uuid
import hmac
import base64
import hashlib
import threading
from typing import List, Dict, Optional
from abc import ABC, abstractmethod
from urllib.parse import urlencode
from datetime import datetime, timezone

# 仅在使用讯飞时导入websocket
try:
    import websocket
    WEBSOCKET_AVAILABLE = True
except ImportError:
    WEBSOCKET_AVAILABLE = False


class LLMInterface(ABC):
    """LLM接口抽象基类"""
    @abstractmethod
    def generate(self, prompt: str, max_tokens: int = 1000, temperature: float = 0.7) -> str:
        pass
    
    @abstractmethod
    def chat(self, messages: List[Dict[str, str]], max_tokens: int = 1000, temperature: float = 0.7) -> str:
        pass


# ==================== MockLLM（纯本地模拟，稳定运行） ====================
class MockLLM(LLMInterface):
    def generate(self, prompt: str, max_tokens: int = 1000, temperature: float = 0.7) -> str:
        prompt_lower = prompt.lower()
        
        if "记忆类型" in prompt or "memory type" in prompt_lower:
            if "2023年" in prompt or "下午" in prompt or "遇到" in prompt or "咖啡店" in prompt:
                return json.dumps({
                    "type": "episodic",
                    "confidence": 0.9,
                    "reasoning": "包含明确的时间和地点信息，是个人经历",
                    "extracted_info": {
                        "temporal_context": "2023年5月15日下午3点",
                        "spatial_context": "星巴克咖啡店",
                        "people": ["李明"]
                    }
                }, ensure_ascii=False)
            elif "是" in prompt and ("语言" in prompt or "位于" in prompt or "属于" in prompt):
                return json.dumps({
                    "type": "semantic",
                    "confidence": 0.85,
                    "reasoning": "包含事实性陈述和定义",
                    "extracted_info": {
                        "entities": [],
                        "relations": []
                    }
                }, ensure_ascii=False)
            elif "如何" in prompt or "步骤" in prompt or "方法" in prompt:
                return json.dumps({
                    "type": "procedural",
                    "confidence": 0.88,
                    "reasoning": "包含步骤性描述和操作方法",
                    "extracted_info": {
                        "steps": []
                    }
                }, ensure_ascii=False)
        
        if "重要性" in prompt or "importance" in prompt_lower:
            return "0.75"
        
        if "相关性" in prompt or "relevance" in prompt_lower:
            if "python" in prompt_lower or "编程" in prompt:
                return "0.85"
            return "0.5"
        
        return json.dumps({
            "type": "unknown",
            "confidence": 0.5,
            "reasoning": "无法明确分类"
        }, ensure_ascii=False)
    
    def chat(self, messages: List[Dict[str, str]], max_tokens: int = 1000, temperature: float = 0.7) -> str:
        last_message = messages[-1]["content"] if messages else ""
        return self.generate(last_message, max_tokens, temperature)


# ==================== 讯飞星火（修复domain为4.0Ultra） ====================
class XinghuoLLM(LLMInterface):
    def __init__(self):
        # 你的密钥（保持不变）
        self.appid = "75714447"
        self.api_key = "79b6bd157e710cac51c22d357d182870"
        self.api_secret = "NjUzMzNjYTE0MTBiODQ0NWVmZTliZDk5"
        
        # 核心修复：domain改为4.0Ultra（匹配讯飞官方有效值）
        self.api_version = "v4.0"
        self.domain = "4.0Ultra"  # 关键修改：替换spark-ultra为4.0Ultra
        self.host = "spark-api.xf-yun.com"
        self.path = f"/{self.api_version}/chat"
        
        if not WEBSOCKET_AVAILABLE:
            raise ImportError("使用讯飞星火需要安装websocket-client: pip install websocket-client")

    def _generate_auth_url(self) -> str:
        """生成v4.0鉴权URL（逻辑不变）"""
        now = datetime.now(timezone.utc)
        date = now.strftime('%a, %d %b %Y %H:%M:%S GMT')
        
        signature_origin = f"host: {self.host}\ndate: {date}\nGET {self.path} HTTP/1.1"
        signature_sha = hmac.new(
            self.api_secret.encode('utf-8'),
            signature_origin.encode('utf-8'),
            digestmod=hashlib.sha256
        ).digest()
        signature_sha_base64 = base64.b64encode(signature_sha).decode('utf-8')
        
        authorization_origin = f'api_key="{self.api_key}", algorithm="hmac-sha256", headers="host date request-line", signature="{signature_sha_base64}"'
        authorization = base64.b64encode(authorization_origin.encode('utf-8')).decode('utf-8')
        
        params = {
            "authorization": authorization,
            "date": date,
            "host": self.host
        }
        return f"wss://{self.host}{self.path}?{urlencode(params)}"

    def _call_websocket_api(self, messages: List[Dict[str, str]], max_tokens: int = 32768, temperature: float = 0.7) -> str:
        """调用讯飞4.0Ultra服务（仅domain参数修改）"""
        result = {"answer": "", "error": None, "finished": False}
        auth_url = self._generate_auth_url()

        def on_error(ws, error):
            nonlocal result
            result["error"] = f"WebSocket错误: {str(error)}"
        
        def on_close(ws, close_status_code, close_msg):
            nonlocal result
            if not result["finished"] and not result["error"]:
                result["error"] = "连接异常关闭"
        
        def on_open(ws):
            """请求体中使用正确的domain: 4.0Ultra"""
            data = {
                "header": {
                    "app_id": self.appid,
                    "uid": str(uuid.uuid4())
                },
                "parameter": {
                    "chat": {
                        "domain": self.domain,  # 使用修复后的4.0Ultra
                        "temperature": temperature,
                        "max_tokens": max_tokens,
                        "top_k": 4,
                        "auditing": "default"
                    }
                },
                "payload": {
                    "message": {
                        "text": messages
                    }
                }
            }
            ws.send(json.dumps(data, ensure_ascii=False))
        
        def on_message(ws, message):
            nonlocal result
            try:
                data = json.loads(message)
                code = data['header']['code']
                
                if code != 0:
                    result["error"] = f"API错误码 {code}: {data['header'].get('message', '未知错误')}"
                    result["finished"] = True
                    ws.close()
                    return
                
                choices = data['payload']['choices']
                result["answer"] += choices['text'][0]['content']
                
                if choices['status'] == 2:
                    result["finished"] = True
                    ws.close()
                    
            except Exception as e:
                result["error"] = f"解析响应失败: {str(e)}"
                result["finished"] = True
                ws.close()
        
        ws = websocket.WebSocketApp(
            auth_url,
            on_message=on_message,
            on_error=on_error,
            on_close=on_close
        )
        ws.on_open = on_open
        
        ws_thread = threading.Thread(
            target=ws.run_forever,
            kwargs={"sslopt": {"cert_reqs": ssl.CERT_NONE}}
        )
        ws_thread.daemon = True
        ws_thread.start()
        
        # 超时处理（60秒）
        timeout = 60
        start_time = time.time()
        while not result["finished"] and not result["error"]:
            if time.time() - start_time > timeout:
                result["error"] = f"调用超时（{timeout}秒）"
                ws.close()
                break
            time.sleep(0.1)
        
        if result["error"]:
            raise RuntimeError(f"讯飞星火（4.0Ultra）调用失败: {result['error']}")
        
        return result["answer"]
    
    def generate(self, prompt: str, max_tokens: int = 32768, temperature: float = 0.7) -> str:
        messages = [{"role": "user", "content": prompt}]
        return self._call_websocket_api(messages, max_tokens, temperature)
    
    def chat(self, messages: List[Dict[str, str]], max_tokens: int = 32768, temperature: float = 0.7) -> str:
        spark_messages = []
        for msg in messages:
            if msg["role"] in ["user", "assistant"]:
                spark_messages.append({"role": msg["role"], "content": msg["content"]})
        
        if not spark_messages:
            return ""
        
        return self._call_websocket_api(spark_messages, max_tokens, temperature)


# ==================== 工厂函数 ====================
def create_llm(provider: str = "mock", **kwargs) -> LLMInterface:
    provider = provider.lower()
    if provider == "mock":
        return MockLLM()
    elif provider in ["xinghuo", "spark", "xfyun"]:
        return XinghuoLLM()
    else:
        raise ValueError(f"不支持的提供者: {provider}。支持: mock, xinghuo")


# ==================== 测试代码 ====================
if __name__ == "__main__":
    # 选项1：测试Mock模式（推荐，无API依赖）
    # print("===== 测试Mock LLM =====")
    # llm = create_llm("mock")
    # result = llm.generate("分析这段文本的记忆类型：2023年5月15日下午3点，我在星巴克咖啡店遇到了大学同学李明。")
    # print(result)

    # 选项2：测试讯飞4.0Ultra服务
    print("===== 测试讯飞Spark Ultra-32K（4.0Ultra） =====")
    try:
        llm = create_llm("xinghuo")
        test_prompt = "分析这段文本的记忆类型：2023年5月15日下午3点，我在星巴克咖啡店遇到了大学同学李明。"
        result = llm.generate(test_prompt, max_tokens=2000)
        print("生成结果：")
        print(result)
    except Exception as e:
        print(f"调用失败：{str(e)}")
        print("\n排查步骤：")
        print("1. 确认domain参数是4.0Ultra（代码已修复）")
        print("2. 检查网络为国内IP（讯飞限制境外访问）")
        print("3. 确认AppID已开通4.0Ultra服务（讯飞控制台→应用→服务管理）")
        print("4. 核对AppID/APIKey/APISecret是否完全匹配")