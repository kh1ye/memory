"""
动态记忆系统使用示例
演示如何使用基于LLM的动态记忆系统
"""

from llm_interface import create_llm
from memory_system import DynamicMemorySystem
from memory_storage import DynamicMemoryStorage
import json


def main():
    print("="*60)
    print("动态记忆系统示例")
    print("="*60)
    
    # 1. 初始化LLM和记忆系统
    print("\n1. 初始化系统...")
    # 使用讯飞星火大模型（已配置API信息）
    llm = create_llm(
        provider="xinghuo",
        appid="75714447",
        api_key="79b6bd157e710cac51c22d357d182870",
        api_secret="NjUzMzNjYTE0MTBiODQ0NWVmZTliZDk5",
        api_version="v4.0",
        domain="4.0Ultra"
    )
    system = DynamicMemorySystem(llm, memory_file="memory/dynamic_memories.json")
    storage = DynamicMemoryStorage(system)
    
    # 2. 存储记忆（使用LLM自动分类和提取）
    print("\n2. 存储记忆...")
    memories_to_store = [
        "2023年5月15日下午3点，我在星巴克咖啡店遇到了大学同学李明，我们一起聊天，感觉很愉快。",
        "Python是一种高级编程语言，由Guido van Rossum在1991年创建。",
        "如何制作一杯拿铁咖啡：首先准备浓缩咖啡，然后加热牛奶至60-70度，接着将热牛奶倒入杯中，最后用奶泡拉花装饰。",
        "昨天在图书馆学习时，我看到一本关于机器学习的书，觉得很有意思。",
        "机器学习是人工智能的一个分支，通过算法让计算机从数据中学习模式。"
    ]
    
    stored_memories = []
    for text in memories_to_store:
        memory = system.store(text)
        if memory:
            stored_memories.append(memory)
            print(f"  ✓ 存储记忆 {memory['id']}: {memory['type']} (置信度: {memory['confidence']:.2f})")
    
    # 3. 检索记忆（基于语义相似度和重要性）
    print("\n3. 检索记忆...")
    queries = ["Python", "咖啡", "学习"]
    for query in queries:
        results = system.retrieve(query, top_k=3)
        print(f"\n  查询: '{query}'")
        print(f"  找到 {len(results)} 条相关记忆:")
        for i, mem in enumerate(results, 1):
            print(f"    {i}. [{mem['type']}] {mem['content'][:50]}...")
    
    # 4. 更新记忆
    print("\n4. 更新记忆...")
    if stored_memories:
        python_memory = next((m for m in stored_memories if "Python" in m.get("content", "")), None)
        if python_memory:
            updated = system.update(
                python_memory['id'],
                "Python是一种高级、解释型编程语言，由Guido van Rossum在1991年创建，特点是语法简洁、易读易写。",
                update_mode="merge"
            )
            print(f"  ✓ 记忆 {updated['id']} 已更新")
            print(f"    更新历史: {len(updated.get('update_history', []))} 次")
    
    # 5. 遗忘策略演示
    print("\n5. 遗忘策略演示...")
    stats_before = system.get_statistics()
    print(f"  遗忘前: {stats_before['total']} 条记忆")
    
    forgotten = system.forget(forget_strategy="low_importance")
    print(f"  已遗忘 {len(forgotten)} 条低重要性记忆")
    
    stats_after = system.get_statistics()
    print(f"  遗忘后: {stats_after['total']} 条记忆")
    
    # 6. 统计信息
    print("\n6. 统计信息:")
    stats = system.get_statistics()
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    
    # 7. 导出记忆
    print("\n7. 导出记忆...")
    output = storage.export_memories(format_type="semantic")
    print(f"  导出格式: semantic")
    print(f"  包含 {len(output.get('episodic_memories', []))} 条情景记忆")
    print(f"  包含 {len(output.get('semantic_memories', []))} 条语义记忆")
    print(f"  包含 {len(output.get('procedural_memories', []))} 条程序记忆")
    
    # 保存到文件
    storage.save_storage_output("memory/dynamic_memory_output.json", format_type="semantic")
    
    # 8. 分析记忆模式
    print("\n8. 记忆模式分析...")
    patterns = storage.analyze_memory_patterns()
    print(f"  平均重要性: {patterns.get('importance_distribution', {}).get('mean', 0):.2f}")
    print(f"  高重要性记忆: {patterns.get('importance_distribution', {}).get('high_importance_count', 0)} 条")
    print(f"  总访问次数: {patterns.get('access_patterns', {}).get('total_access', 0)}")
    
    print("\n" + "="*60)
    print("示例完成！")
    print("="*60)


if __name__ == "__main__":
    main()
