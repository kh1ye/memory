"""
讯飞星火API连接测试脚本
用于快速验证API配置是否正确
"""

from llm_interface import create_llm

# 你的讯飞星火API配置
APPID = "75714447"
API_KEY = "79b6bd157e710cac51c22d357d182870"
API_SECRET = "NjUzMzNjYTE0MTBiODQ0NWVmZTliZDk5"

def test_connection():
    """测试讯飞星火API连接"""
    print("=" * 60)
    print("🔍 讯飞星火API连接测试")
    print("=" * 60)
    print(f"\n📋 配置信息:")
    print(f"   APPID: {APPID}")
    print(f"   API_KEY: {API_KEY[:20]}...")
    print(f"   API_SECRET: {API_SECRET[:20]}...")
    
    # 1. 初始化LLM
    print("\n1️⃣ 初始化讯飞星火LLM...")
    try:
        llm = create_llm(
            provider="xinghuo",
            appid=APPID,
            api_key=API_KEY,
            api_secret=API_SECRET,
            api_version="v3.5",
            domain="generalv3.5"
        )
        print("✅ LLM初始化成功")
    except Exception as e:
        print(f"❌ LLM初始化失败: {e}")
        print("\n💡 请检查:")
        print("   1. 是否安装了websocket-client: pip install websocket-client")
        print("   2. API配置信息是否正确")
        return False
    
    # 2. 测试简单对话
    print("\n2️⃣ 测试简单对话...")
    try:
        response = llm.generate("你好，请回答：1+1等于几？")
        print(f"✅ 测试成功！")
        print(f"\n📝 模型响应:")
        print(f"   {response}")
        return True
    except Exception as e:
        print(f"❌ 对话测试失败: {e}")
        print("\n💡 可能的原因:")
        print("   1. API密钥无效或已过期")
        print("   2. 网络无法访问 spark-api.xf-yun.com")
        print("   3. 账户余额不足或免费额度已用完")
        print("   4. API版本不匹配（尝试更改 api_version 参数）")
        return False
    
    # 3. 测试多轮对话
    print("\n3️⃣ 测试多轮对话...")
    try:
        messages = [
            {"role": "user", "content": "我叫张三"},
            {"role": "assistant", "content": "好的，我记住了你叫张三。"},
            {"role": "user", "content": "请重复我的名字"}
        ]
        response = llm.chat(messages)
        print(f"✅ 多轮对话测试成功！")
        print(f"\n📝 模型响应:")
        print(f"   {response}")
        return True
    except Exception as e:
        print(f"❌ 多轮对话测试失败: {e}")
        return False


def main():
    """主函数"""
    success = test_connection()
    
    print("\n" + "=" * 60)
    if success:
        print("🎉 所有测试通过！API配置正确，可以使用。")
        print("\n📖 下一步:")
        print("   运行 python quick_start_xinghuo.py 开始使用动态记忆系统")
    else:
        print("⚠️  测试失败，请检查配置和网络连接")
    print("=" * 60)


if __name__ == "__main__":
    main()
