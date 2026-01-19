"""
动态记忆系统 - 基于大模型的学习式记忆管理
核心思想：
1. Prompt即学习结果：通过模型学习得出Prompt，而非人工硬编码
2. 拒绝机械化抽取：记忆格式通过学习机制自然形成，模仿人类注意偏差
3. 动态记忆系统：包含存储、检索、遗忘、更新四大动态过程
"""

import json
import time
from typing import List, Dict, Optional, Callable, Tuple
from enum import Enum
from datetime import datetime

try:
    import numpy as np
except ImportError:
    # 如果numpy不可用，使用内置统计
    class np:
        @staticmethod
        def mean(values):
            return sum(values) / len(values) if values else 0


# 导入LLM接口
try:
    from llm_interface import LLMInterface, create_llm
except ImportError:
    # 如果llm_interface不存在，使用简单的后备实现
    class LLMInterface:
        """占位接口"""
        pass
    
    def create_llm(provider="mock", **kwargs):
        # 简单后备，实际应该导入MockLLM
        from llm_interface import MockLLM
        return MockLLM()


class MemoryType(Enum):
    """记忆类型枚举"""
    EPISODIC = "episodic"  # 情景记忆
    SEMANTIC = "semantic"  # 语义记忆
    PROCEDURAL = "procedural"  # 程序记忆




class PromptLearner:
    """
    Prompt学习器
    通过LLM来学习和优化Prompt，而不是人工硬编码
    """
    
    def __init__(self, llm: LLMInterface):
        """
        初始化Prompt学习器
        
        Args:
            llm: LLM接口
        """
        self.llm = llm
        self.learned_prompts = {
            "memory_classification": self._initial_classification_prompt(),
            "memory_extraction": self._initial_extraction_prompt(),
            "memory_importance": self._initial_importance_prompt(),
            "memory_updating": self._initial_updating_prompt()
        }
        self.prompt_history = []  # 记录Prompt演化历史
    
    def _initial_classification_prompt(self) -> str:
        """初始记忆分类Prompt"""
        return """你是一个记忆系统分析专家。请分析以下文本，判断它属于哪种记忆类型：
1. 情景记忆（Episodic）：特定时间、地点发生的具体事件，包含个人经历
2. 语义记忆（Semantic）：客观事实、概念、知识和规律
3. 程序记忆（Procedural）：如何做某事的技能、步骤和流程

文本：{text}

请以JSON格式返回分析结果，包含：
- type: 记忆类型（episodic/semantic/procedural）
- confidence: 置信度（0-1）
- reasoning: 判断理由
- extracted_info: 提取的关键信息"""
    
    def _initial_extraction_prompt(self) -> str:
        """初始记忆提取Prompt"""
        return """请从以下文本中提取记忆的关键信息。不要使用硬编码的规则，而是根据文本的自然语义理解来提取：

文本：{text}
记忆类型：{memory_type}

请以JSON格式返回，格式应该自然形成，而不是强制满足某个模板。根据文本内容的特点来组织信息。"""
    
    def _initial_importance_prompt(self) -> str:
        """初始重要性评估Prompt（模仿Attention Bias）"""
        return """模仿人类的注意力偏差（Attention Bias），评估以下记忆的重要性。
人类会自然关注：
- 情感强度高的经历
- 与当前目标相关的信息
- 重复出现或频繁访问的内容
- 新奇的、突破性的知识

记忆内容：{memory}
访问历史：{access_history}
上下文目标：{context_goal}

请评估重要性分数（0-1），并说明理由。"""
    
    def _initial_updating_prompt(self) -> str:
        """初始记忆更新Prompt"""
        return """记忆需要动态更新。当新信息与旧记忆冲突或补充时，应该如何更新？

旧记忆：{old_memory}
新信息：{new_info}

请提供更新后的记忆，并说明更新的理由和方式。"""
    
    def optimize_prompt(self, prompt_name: str, examples: List[Dict], feedback: Optional[Dict] = None) -> str:
        """
        通过示例和反馈优化Prompt
        
        Args:
            prompt_name: Prompt名称
            examples: 示例列表（包含输入和期望输出）
            feedback: 反馈信息（可选）
            
        Returns:
            优化后的Prompt
        """
        optimization_prompt = f"""你是一个Prompt优化专家。当前Prompt效果不理想，请根据以下示例和反馈来优化它。

当前Prompt：
{self.learned_prompts[prompt_name]}

示例：
{json.dumps(examples, ensure_ascii=False, indent=2)}

{"反馈：" + json.dumps(feedback, ensure_ascii=False) if feedback else ""}

请提供优化后的Prompt，使其能更好地完成任务。只返回优化后的Prompt内容，不要其他说明。"""
        
        optimized = self.llm.generate(optimization_prompt)
        self.learned_prompts[prompt_name] = optimized
        self.prompt_history.append({
            "name": prompt_name,
            "timestamp": datetime.now().isoformat(),
            "old_prompt": self.learned_prompts[prompt_name],
            "new_prompt": optimized
        })
        return optimized
    
    def get_prompt(self, prompt_name: str) -> str:
        """获取指定的Prompt"""
        return self.learned_prompts.get(prompt_name, "")


class DynamicMemorySystem:
    """
    动态记忆系统
    实现存储、检索、遗忘、更新四大动态过程
    """
    
    def __init__(self, llm: LLMInterface, memory_file: str = "dynamic_memories.json"):
        """
        初始化动态记忆系统
        
        Args:
            llm: LLM接口
            memory_file: 记忆存储文件路径
        """
        self.llm = llm
        self.memory_file = memory_file
        self.prompt_learner = PromptLearner(llm)
        self.memories: List[Dict] = []
        self.access_history: Dict[int, List[float]] = {}  # 记忆访问历史 {memory_id: [timestamps]}
        self.importance_scores: Dict[int, float] = {}  # 记忆重要性分数
        self.load_memories()
    
    def load_memories(self):
        """从文件加载记忆"""
        try:
            with open(self.memory_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.memories = data.get('memories', [])
                self.access_history = {int(k): v for k, v in data.get('access_history', {}).items()}
                self.importance_scores = {int(k): float(v) for k, v in data.get('importance_scores', {}).items()}
        except FileNotFoundError:
            self.memories = []
            self.access_history = {}
            self.importance_scores = {}
        except Exception as e:
            print(f"加载记忆时出错: {e}")
            self.memories = []
    
    def save_memories(self):
        """保存记忆到文件"""
        data = {
            'memories': self.memories,
            'access_history': self.access_history,
            'importance_scores': self.importance_scores,
            'last_updated': datetime.now().isoformat()
        }
        with open(self.memory_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    # ============ 核心四大动态过程 ============
    
    def store(self, text: str, context: Optional[Dict] = None) -> Dict:
        """
        1. 存储：使用LLM理解和存储新记忆
        
        Args:
            text: 要存储的文本
            context: 上下文信息（可选）
            
        Returns:
            存储的记忆对象
        """
        # 使用学习到的Prompt进行分类和提取
        classification_prompt = self.prompt_learner.get_prompt("memory_classification").format(text=text)
        classification_result = self.llm.generate(classification_prompt)
        
        try:
            classification_data = json.loads(classification_result)
        except json.JSONDecodeError:
            # 如果LLM返回的不是JSON，尝试提取JSON部分
            import re
            json_match = re.search(r'\{.*\}', classification_result, re.DOTALL)
            if json_match:
                classification_data = json.loads(json_match.group())
            else:
                classification_data = {"type": "unknown", "confidence": 0.5}
        
        memory_type = classification_data.get("type", "unknown")
        confidence = classification_data.get("confidence", 0.5)
        
        # 如果置信度太低，不存储
        if confidence < 0.3:
            return None
        
        # 提取记忆信息
        extraction_prompt = self.prompt_learner.get_prompt("memory_extraction").format(
            text=text,
            memory_type=memory_type
        )
        extraction_result = self.llm.generate(extraction_prompt)
        
        try:
            extracted_info = json.loads(extraction_result)
        except json.JSONDecodeError:
            extracted_info = {"raw_text": text}
        
        # 创建记忆对象
        memory_id = len(self.memories) + 1
        memory = {
            "id": memory_id,
            "type": memory_type,
            "content": text,
            "extracted_info": extracted_info,
            "confidence": confidence,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "access_count": 0,
            "importance_score": 0.5,  # 初始重要性
            "context": context or {}
        }
        
        # 计算初始重要性
        memory["importance_score"] = self._calculate_importance(memory, context)
        self.importance_scores[memory_id] = memory["importance_score"]
        self.access_history[memory_id] = []
        
        self.memories.append(memory)
        self.save_memories()
        
        return memory
    
    def retrieve(self, query: str, top_k: int = 5, memory_type: Optional[str] = None) -> List[Dict]:
        """
        2. 检索：基于语义相似度和重要性检索记忆
        
        Args:
            query: 查询文本
            top_k: 返回前k个结果
            memory_type: 限制记忆类型（可选）
            
        Returns:
            检索到的记忆列表
        """
        # 使用LLM计算相关性
        relevance_scores = []
        
        for memory in self.memories:
            if memory_type and memory.get("type") != memory_type:
                continue
            
            # 构建相关性评估Prompt
            relevance_prompt = f"""评估以下查询与记忆的相关性（0-1分数）：

查询：{query}
记忆内容：{memory.get("content", "")}
记忆类型：{memory.get("type", "")}

请只返回一个0-1之间的浮点数分数，不要其他内容。"""
            
            relevance_result = self.llm.generate(relevance_prompt)
            try:
                relevance_score = float(relevance_result.strip())
            except (ValueError, AttributeError):
                # 如果无法解析，使用简单的关键词匹配作为后备
                query_words = set(query.lower().split())
                content_words = set(memory.get("content", "").lower().split())
                relevance_score = len(query_words & content_words) / max(len(query_words), 1) * 0.5
            
            # 综合相关性、重要性和访问频率
            importance = self.importance_scores.get(memory["id"], 0.5)
            access_count = memory.get("access_count", 0)
            
            # 计算综合分数（加权平均）
            final_score = (
                0.5 * relevance_score +
                0.3 * importance +
                0.2 * min(access_count / 10, 1.0)  # 访问频率（归一化）
            )
            
            relevance_scores.append((memory, final_score))
        
        # 按分数排序
        relevance_scores.sort(key=lambda x: x[1], reverse=True)
        
        # 更新访问历史
        current_time = time.time()
        for memory, _ in relevance_scores[:top_k]:
            memory_id = memory["id"]
            memory["access_count"] = memory.get("access_count", 0) + 1
            if memory_id not in self.access_history:
                self.access_history[memory_id] = []
            self.access_history[memory_id].append(current_time)
        
        self.save_memories()
        
        return [memory for memory, _ in relevance_scores[:top_k]]
    
    def forget(self, memory_id: Optional[int] = None, forget_strategy: str = "low_importance") -> List[Dict]:
        """
        3. 遗忘：根据策略遗忘不重要的记忆
        
        Args:
            memory_id: 指定要遗忘的记忆ID（如果为None，则根据策略自动选择）
            forget_strategy: 遗忘策略
                - "low_importance": 遗忘重要性低的记忆
                - "old_unused": 遗忘旧的、很少访问的记忆
                - "low_relevance": 遗忘与当前上下文不相关的记忆
                
        Returns:
            被遗忘的记忆列表
        """
        forgotten = []
        
        if memory_id is not None:
            # 删除指定记忆
            memory = next((m for m in self.memories if m["id"] == memory_id), None)
            if memory:
                self.memories.remove(memory)
                forgotten.append(memory)
                if memory_id in self.importance_scores:
                    del self.importance_scores[memory_id]
                if memory_id in self.access_history:
                    del self.access_history[memory_id]
        else:
            # 根据策略选择要遗忘的记忆
            candidates = []
            
            for memory in self.memories:
                if forget_strategy == "low_importance":
                    importance = self.importance_scores.get(memory["id"], 0.5)
                    if importance < 0.3:
                        candidates.append((memory, importance))
                
                elif forget_strategy == "old_unused":
                    access_count = memory.get("access_count", 0)
                    created_at = datetime.fromisoformat(memory.get("created_at", datetime.now().isoformat()))
                    age_days = (datetime.now() - created_at).days
                    if access_count < 2 and age_days > 30:
                        candidates.append((memory, 1.0 / (age_days + 1)))
            
            # 选择最不重要的记忆删除（保留至少80%的记忆）
            candidates.sort(key=lambda x: x[1])
            forget_count = max(0, len(candidates) - int(len(self.memories) * 0.8))
            
            for memory, _ in candidates[:forget_count]:
                self.memories.remove(memory)
                forgotten.append(memory)
                memory_id = memory["id"]
                if memory_id in self.importance_scores:
                    del self.importance_scores[memory_id]
                if memory_id in self.access_history:
                    del self.access_history[memory_id]
        
        if forgotten:
            self.save_memories()
        
        return forgotten
    
    def update(self, memory_id: int, new_info: str, update_mode: str = "merge") -> Dict:
        """
        4. 更新：动态更新已有记忆
        
        Args:
            memory_id: 记忆ID
            new_info: 新信息
            update_mode: 更新模式
                - "merge": 合并新信息
                - "replace": 替换旧记忆
                - "refine": 精炼和修正
                
        Returns:
            更新后的记忆对象
        """
        memory = next((m for m in self.memories if m["id"] == memory_id), None)
        if not memory:
            raise ValueError(f"记忆 {memory_id} 不存在")
        
        old_content = memory.get("content", "")
        
        # 使用LLM决定如何更新
        updating_prompt = self.prompt_learner.get_prompt("memory_updating").format(
            old_memory=old_content,
            new_info=new_info
        )
        
        update_result = self.llm.generate(updating_prompt)
        
        try:
            update_data = json.loads(update_result)
            updated_content = update_data.get("updated_content", new_info)
            update_reason = update_data.get("reasoning", "")
        except json.JSONDecodeError:
            # 如果无法解析，根据模式合并
            if update_mode == "merge":
                updated_content = f"{old_content}\n{new_info}"
            elif update_mode == "replace":
                updated_content = new_info
            else:  # refine
                updated_content = new_info
            update_reason = f"使用{update_mode}模式更新"
        
        # 更新记忆
        memory["content"] = updated_content
        memory["updated_at"] = datetime.now().isoformat()
        memory["update_history"] = memory.get("update_history", [])
        memory["update_history"].append({
            "timestamp": datetime.now().isoformat(),
            "new_info": new_info,
            "mode": update_mode,
            "reasoning": update_reason
        })
        
        # 重新计算重要性
        memory["importance_score"] = self._calculate_importance(memory, memory.get("context"))
        self.importance_scores[memory_id] = memory["importance_score"]
        
        self.save_memories()
        
        return memory
    
    # ============ 辅助方法 ============
    
    def _calculate_importance(self, memory: Dict, context: Optional[Dict] = None) -> float:
        """
        计算记忆重要性（模仿Attention Bias）
        
        Args:
            memory: 记忆对象
            context: 上下文信息
            
        Returns:
            重要性分数（0-1）
        """
        importance_prompt = self.prompt_learner.get_prompt("memory_importance").format(
            memory=memory.get("content", ""),
            access_history=str(self.access_history.get(memory["id"], [])),
            context_goal=json.dumps(context, ensure_ascii=False) if context else "无特定目标"
        )
        
        importance_result = self.llm.generate(importance_prompt)
        
        try:
            # 尝试提取数字
            import re
            score_match = re.search(r'0?\.\d+|1\.0|0', importance_result)
            if score_match:
                return float(score_match.group())
        except:
            pass
        
        # 默认重要性计算（基于一些启发式规则）
        base_score = 0.5
        confidence = memory.get("confidence", 0.5)
        memory_type = memory.get("type", "unknown")
        
        # 不同类型的默认重要性
        type_weights = {
            "episodic": 0.6,  # 情景记忆通常更重要（个人经历）
            "semantic": 0.5,
            "procedural": 0.7  # 程序记忆通常更重要（技能）
        }
        
        return base_score * confidence * type_weights.get(memory_type, 0.5)
    
    def get_statistics(self) -> Dict:
        """获取记忆统计信息"""
        type_counts = {}
        for memory in self.memories:
            mem_type = memory.get("type", "unknown")
            type_counts[mem_type] = type_counts.get(mem_type, 0) + 1
        
        return {
            "total": len(self.memories),
            "by_type": type_counts,
            "average_importance": np.mean(list(self.importance_scores.values())) if self.importance_scores else 0,
            "total_access_count": sum(m.get("access_count", 0) for m in self.memories)
        }
    
    def search_memories(self, keyword: str) -> List[Dict]:
        """关键词搜索（向后兼容）"""
        return self.retrieve(keyword, top_k=10)


if __name__ == "__main__":
    # 示例使用 - 使用讯飞星火大模型
    llm = create_llm(
        provider="xinghuo",
        appid="75714447",
        api_key="79b6bd157e710cac51c22d357d182870",
        api_secret="NjUzMzNjYTE0MTBiODQ0NWVmZTliZDk5",
        api_version="v3.5",
        domain="generalv3.5"
    )
    system = DynamicMemorySystem(llm)
    
    # 存储记忆
    print("存储记忆...")
    memory1 = system.store("2023年5月15日下午3点，我在星巴克咖啡店遇到了大学同学李明。")
    print(f"存储记忆: {memory1['id']} - {memory1['type']}")
    
    memory2 = system.store("Python是一种高级编程语言，由Guido van Rossum创建。")
    print(f"存储记忆: {memory2['id']} - {memory2['type']}")
    
    # 检索记忆
    print("\n检索记忆...")
    results = system.retrieve("Python", top_k=3)
    print(f"检索到 {len(results)} 条相关记忆")
    
    # 更新记忆
    print("\n更新记忆...")
    updated = system.update(memory2['id'], "Python是一种高级编程语言，由Guido van Rossum在1991年创建。")
    print(f"记忆已更新: {updated['id']}")
    
    # 统计信息
    print("\n统计信息:")
    stats = system.get_statistics()
    print(json.dumps(stats, ensure_ascii=False, indent=2))
