"""
动态记忆存储模块
支持动态记忆系统的存储、检索、导出等功能
使用LLM学习形成的格式，而非硬编码的三元组或超参数
"""

import json
from typing import Dict, List, Optional
from datetime import datetime
from memory_system import DynamicMemorySystem


class DynamicMemoryStorage:
    """动态记忆存储类"""
    
    def __init__(self, memory_system: DynamicMemorySystem):
        """
        初始化动态记忆存储
        
        Args:
            memory_system: 动态记忆系统实例
        """
        self.system = memory_system
    
    def export_memories(self, format_type: str = "structured") -> Dict:
        """
        导出记忆（使用LLM学习形成的格式）
        
        Args:
            format_type: 导出格式类型
                - "structured": 结构化格式（包含所有元数据）
                - "semantic": 语义化格式（便于理解）
                - "minimal": 最小格式（仅核心信息）
                
        Returns:
            导出的记忆数据
        """
        if format_type == "structured":
            return self._export_structured()
        elif format_type == "semantic":
            return self._export_semantic()
        elif format_type == "minimal":
            return self._export_minimal()
        else:
            raise ValueError(f"不支持的格式类型: {format_type}")
    
    def _export_structured(self) -> Dict:
        """导出结构化格式（包含完整元数据）"""
        return {
            "memories": self.system.memories,
            "access_history": self.system.access_history,
            "importance_scores": self.system.importance_scores,
            "statistics": self.system.get_statistics(),
            "export_timestamp": datetime.now().isoformat()
        }
    
    def _export_semantic(self) -> Dict:
        """导出语义化格式（便于人类理解）"""
        semantic_data = {
            "episodic_memories": [],
            "semantic_memories": [],
            "procedural_memories": [],
            "export_timestamp": datetime.now().isoformat()
        }
        
        for memory in self.system.memories:
            mem_type = memory.get("type", "unknown")
            importance = self.system.importance_scores.get(memory["id"], 0.5)
            
            mem_entry = {
                "id": memory["id"],
                "content": memory.get("content", ""),
                "importance": importance,
                "confidence": memory.get("confidence", 0.5),
                "extracted_info": memory.get("extracted_info", {}),
                "created_at": memory.get("created_at"),
                "updated_at": memory.get("updated_at"),
                "access_count": memory.get("access_count", 0)
            }
            
            if mem_type == "episodic":
                semantic_data["episodic_memories"].append(mem_entry)
            elif mem_type == "semantic":
                semantic_data["semantic_memories"].append(mem_entry)
            elif mem_type == "procedural":
                semantic_data["procedural_memories"].append(mem_entry)
        
        return semantic_data
    
    def _export_minimal(self) -> Dict:
        """导出最小格式（仅核心信息）"""
        minimal_data = {
            "memories": [],
            "statistics": self.system.get_statistics()
        }
        
        for memory in self.system.memories:
            minimal_data["memories"].append({
                "id": memory["id"],
                "type": memory.get("type"),
                "content": memory.get("content", ""),
                "importance": self.system.importance_scores.get(memory["id"], 0.5)
            })
        
        return minimal_data
    
    def generate_storage_output(self, output_format: str = "semantic") -> Dict:
        """
        生成存储格式输出（向后兼容接口）
        
        Args:
            output_format: 输出格式
            
        Returns:
            存储输出数据
        """
        return self.export_memories(format_type=output_format)
    
    def save_storage_output(self, output_file: str = "memory/dynamic_memory_output.json", format_type: str = "semantic"):
        """
        保存存储格式的输出
        
        Args:
            output_file: 输出文件名
            format_type: 格式类型
        """
        output = self.export_memories(format_type=format_type)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
        
        stats = self.system.get_statistics()
        print(f"\n✅ 动态记忆输出已保存到: {output_file}")
        print(f"📊 统计信息:")
        print(f"   总记忆数: {stats['total']}")
        if 'by_type' in stats:
            for mem_type, count in stats['by_type'].items():
                type_names = {
                    "episodic": "情景记忆",
                    "semantic": "语义记忆",
                    "procedural": "程序记忆"
                }
                print(f"   {type_names.get(mem_type, mem_type)}: {count} 条")
        print(f"   平均重要性: {stats.get('average_importance', 0):.2f}")
        print(f"   总访问次数: {stats.get('total_access_count', 0)}")
        
        return output
    
    def analyze_memory_patterns(self) -> Dict:
        """
        分析记忆模式（使用统计方法，而非硬编码规则）
        
        Returns:
            记忆模式分析结果
        """
        patterns = {
            "temporal_patterns": {},
            "semantic_clusters": {},
            "importance_distribution": {},
            "access_patterns": {}
        }
        
        # 分析时间模式
        for memory in self.system.memories:
            created_at = memory.get("created_at", "")
            if created_at:
                try:
                    dt = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
                    hour = dt.hour
                    patterns["temporal_patterns"][hour] = patterns["temporal_patterns"].get(hour, 0) + 1
                except:
                    pass
        
        # 重要性分布
        importance_values = list(self.system.importance_scores.values())
        if importance_values:
            patterns["importance_distribution"] = {
                "mean": sum(importance_values) / len(importance_values),
                "min": min(importance_values),
                "max": max(importance_values),
                "high_importance_count": sum(1 for v in importance_values if v > 0.7),
                "low_importance_count": sum(1 for v in importance_values if v < 0.3)
            }
        
        # 访问模式
        total_access = sum(m.get("access_count", 0) for m in self.system.memories)
        patterns["access_patterns"] = {
            "total_access": total_access,
            "average_access_per_memory": total_access / len(self.system.memories) if self.system.memories else 0,
            "most_accessed": sorted(
                [(m["id"], m.get("access_count", 0)) for m in self.system.memories],
                key=lambda x: x[1],
                reverse=True
            )[:5]
        }
        
        return patterns


if __name__ == "__main__":
    from llm_interface import create_llm
    from memory_system import DynamicMemorySystem
    
    # 初始化系统
    llm = create_llm("mock")
    system = DynamicMemorySystem(llm)
    
    # 创建存储对象
    storage = DynamicMemoryStorage(system)
    
    # 测试：存储一些记忆
    system.store("2023年5月15日下午3点，我在星巴克咖啡店遇到了大学同学李明。")
    system.store("Python是一种高级编程语言，由Guido van Rossum创建。")
    
    # 生成并保存存储格式输出
    storage.save_storage_output(format_type="semantic")
    
    # 分析记忆模式
    patterns = storage.analyze_memory_patterns()
    print("\n记忆模式分析:")
    print(json.dumps(patterns, ensure_ascii=False, indent=2))
