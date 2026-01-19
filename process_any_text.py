"""
处理任意文本 - 将任意一段话转换成三种记忆（使用动态记忆系统）
"""

import json
from llm_interface import create_llm
from memory_system import DynamicMemorySystem
from memory_storage import DynamicMemoryStorage


def process_any_text(text: str, output_file: str = "memory/output_from_text.json"):
    """
    处理任意文本，提取三种记忆类型（使用动态记忆系统）
    
    Args:
        text: 输入的文本
        output_file: 输出文件名
    """
    # 初始化讯飞星火LLM和动态记忆系统
    llm = create_llm(
        provider="xinghuo",
        appid="75714447",
        api_key="79b6bd157e710cac51c22d357d182870",
        api_secret="NjUzMzNjYTE0MTBiODQ0NWVmZTliZDk5",
        api_version="v4.0",
        domain="4.0Ultra"
    )
    system = DynamicMemorySystem(llm)
    storage = DynamicMemoryStorage(system)
    
    # 按句子分割文本并存储
    import re
    sentences = re.split(r'([。！？.!?])', text)
    current_sentence = ""
    
    for part in sentences:
        current_sentence += part
        if part in ['。', '！', '？', '.', '!', '?'] and current_sentence.strip():
            content = current_sentence.strip()
            if len(content) > 5:  # 过滤太短的片段
                system.store(content)
            current_sentence = ""
    
    # 处理最后一段
    if current_sentence.strip() and len(current_sentence.strip()) > 5:
        system.store(current_sentence.strip())
    
    # 生成输出
    stats = system.get_statistics()
    output_data = {
        "input_text": text,
        "episodic_memories": [],
        "semantic_memories": [],
        "procedural_memories": [],
        "statistics": stats
    }
    
    # 按类型收集记忆
    for memory in system.memories:
        mem_type = memory.get("type", "unknown")
        mem_entry = {
            "id": memory.get("id"),
            "content": memory.get("content"),
            "confidence": memory.get("confidence", 0.5),
            "importance": system.importance_scores.get(memory["id"], 0.5),
            "identification_method": "llm_classification",
            "extracted_info": memory.get("extracted_info", {})
        }
        
        if mem_type == "episodic":
            output_data["episodic_memories"].append(mem_entry)
        elif mem_type == "semantic":
            output_data["semantic_memories"].append(mem_entry)
        elif mem_type == "procedural":
            output_data["procedural_memories"].append(mem_entry)
    
    # 保存输出
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    print(f"\n✅ 处理完成！")
    print(f"📝 输入文本长度: {len(text)} 字符")
    print(f"📊 统计信息:")
    print(f"   情景记忆: {output_data['statistics']['episodic']} 条")
    print(f"   语义记忆: {output_data['statistics']['semantic']} 条")
    print(f"   程序记忆: {output_data['statistics']['procedural']} 条")
    print(f"   总计: {output_data['statistics']['total']} 条")
    print(f"💾 结果已保存到: {output_file}")
    
    return output_data


if __name__ == "__main__":
    # 示例：处理任意文本
    sample_text = """
    今天早上8点，我在办公室和同事讨论了一个新项目。机器学习是人工智能的核心技术之一，它通过算法从数据中学习模式。
    如何准备一个会议：首先确定会议主题和目标，然后邀请相关参与者，准备会议材料，最后安排会议时间和地点。
    昨天下午3点，我在图书馆阅读了一本关于深度学习的书籍，学到了很多新知识。神经网络是深度学习的基础架构。
    """
    
    print("=" * 60)
    print("处理任意文本示例")
    print("=" * 60)
    print(f"\n输入文本:\n{sample_text}\n")
    
    result = process_any_text(sample_text.strip())
    
    print("\n" + "=" * 60)
    print("提取的记忆详情")
    print("=" * 60)
    
    if result["episodic_memories"]:
        print("\n【情景记忆】")
        for mem in result["episodic_memories"]:
            print(f"  - {mem['content']}")
    
    if result["semantic_memories"]:
        print("\n【语义记忆】")
        for mem in result["semantic_memories"]:
            print(f"  - {mem['content']}")
    
    if result["procedural_memories"]:
        print("\n【程序记忆】")
        for mem in result["procedural_memories"]:
            print(f"  - {mem['content']}")
            if mem.get('extracted_steps'):
                print("    步骤:")
                for i, step in enumerate(mem['extracted_steps'], 1):
                    print(f"      {i}. {step}")
