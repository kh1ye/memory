"""
讯飞星火快速开始脚本
最简单的使用示例
"""

from llm_interface import create_llm
from memory_system import DynamicMemorySystem

# 配置你的讯飞星火API信息
XINGHUO_CONFIG = {
    "appid": "75714447",
    "api_key": "79b6bd157e710cac51c22d357d182870",
    "api_secret": "NjUzMzNjYTE0MTBiODQ0NWVmZTliZDk5"
}

def main():
    print("🚀 讯飞星火动态记忆系统 - 快速开始")
    print("=" * 50)
    
    # 1. 初始化
    print("\n1️⃣ 初始化讯飞星火LLM...")
    llm = create_llm(provider="xinghuo", **XINGHUO_CONFIG)
    system = DynamicMemorySystem(llm)
    print("✅ 初始化成功！")
    
    # 2. 测试连接
    print("\n2️⃣ 测试LLM连接...")
    try:
        response = llm.generate("请回答：你好")
        print(f"✅ LLM响应正常: {response[:50]}...")
    except Exception as e:
        print(f"❌ 连接失败: {e}")
        print("\n💡 提示：")
        print("   - 请检查API密钥是否正确")
        print("   - 请确保网络可以访问 spark-api.xf-yun.com")
        return
    
    # 3. 存储记忆
    print("\n3️⃣ 存储记忆示例...")
    memory = system.store("2024年1月1日，我在北京的天安门广场观看了升旗仪式，心情非常激动。")
    if memory:
        print(f"✅ 记忆已存储: ID={memory['id']}, 类型={memory['type']}, 置信度={memory['confidence']:.2f}")
    
    # 4. 检索记忆
    print("\n4️⃣ 检索记忆示例...")
    results = system.retrieve("北京", top_k=2)
    print(f"✅ 检索到 {len(results)} 条相关记忆")
    for i, mem in enumerate(results, 1):
        print(f"   {i}. {mem['content'][:60]}...")
    
    # 5. 统计信息
    print("\n5️⃣ 统计信息...")
    stats = system.get_statistics()
    print(f"✅ 总记忆数: {stats['total']}")
    print(f"✅ 按类型分布: {stats['by_type']}")
    
    print("\n" + "=" * 50)
    print("🎉 快速开始完成！")
    print("\n📖 更多示例请查看 example_xinghuo_usage.py")

if __name__ == "__main__":
    main()
