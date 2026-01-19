# 动态记忆系统

基于大模型的学习式记忆管理系统，实现真正的动态记忆（存储、检索、遗忘、更新）。

## 核心思想

### 1. Prompt即学习结果
- ❌ **旧方法**：人工硬编码Prompt和规则
- ✅ **新方法**：通过大模型学习和优化Prompt，使Prompt本身成为系统的学习产物

### 2. 拒绝机械化抽取
- ❌ **旧方法**：使用固定超参数、正则表达式、关键词匹配来硬性切割记忆
- ✅ **新方法**：通过LLM的自然语义理解来形成记忆格式，模仿人类的注意偏差（Attention Bias）

### 3. 动态 vs 静态
- ❌ **旧方法**：静态的"向量数据库+百科全书"，缺乏演化能力
- ✅ **新方法**：动态记忆系统，包含四大核心过程：
  - **存储（Store）**：使用LLM理解并存储新记忆
  - **检索（Retrieve）**：基于语义相似度和重要性检索
  - **遗忘（Forget）**：根据策略动态遗忘不重要的记忆
  - **更新（Update）**：智能更新已有记忆，支持合并、替换、精炼

## 系统架构

```
动态记忆系统
├── LLM接口层 (llm_interface.py)
│   ├── OpenAILLM - OpenAI API实现
│   ├── AnthropicLLM - Anthropic Claude API实现
│   └── MockLLM - 模拟LLM（用于测试）
│
├── Prompt学习层 (memory_system.py -> PromptLearner)
│   ├── 记忆分类Prompt
│   ├── 记忆提取Prompt
│   ├── 重要性评估Prompt
│   └── 记忆更新Prompt
│
├── 动态记忆核心 (memory_system.py -> DynamicMemorySystem)
│   ├── 存储（Store）
│   ├── 检索（Retrieve）
│   ├── 遗忘（Forget）
│   └── 更新（Update）
│
└── 存储接口层 (memory_storage.py)
    ├── 导出功能
    ├── 格式转换
    └── 模式分析
```

## 文件说明

### 核心文件
- `llm_interface.py` - LLM接口实现，支持多种后端（讯飞星火、OpenAI、Anthropic、Mock）
- `memory_system.py` - 动态记忆系统核心，包含Prompt学习和四大动态过程
- `memory_storage.py` - 动态记忆存储和导出功能
- `example_xinghuo_usage.py` - 讯飞星火使用示例（推荐）
- `example_dynamic_usage.py` - 通用使用示例脚本

### 数据文件
- `dynamic_memories.json` - 动态记忆存储文件（自动生成）
- `dynamic_memory_output.json` - 导出的记忆数据

## 安装依赖

```bash
# 基础依赖（必需）
pip install numpy websocket-client

# 使用讯飞星火（已包含在基础依赖中）
# websocket-client已包含

# 使用OpenAI（可选）
# pip install openai

# 使用Anthropic（可选）
# pip install anthropic
```

## 快速开始

### 基本使用

```python
from llm_interface import create_llm
from memory_system import DynamicMemorySystem
from memory_storage import DynamicMemoryStorage

# 1. 初始化LLM（推荐使用讯飞星火大模型）
llm = create_llm(
    provider="xinghuo",
    appid="你的APPID",
    api_key="你的APIKey",
    api_secret="你的APISecret",
    api_version="v3.5",  # 可选：v3.1, v3.5, v4.0
    domain="generalv3.5"  # 可选：generalv3, generalv3.5, generalv4
)

# 或使用Mock LLM（测试用，不需要API密钥）
# llm = create_llm("mock")

# 或使用OpenAI（需要API密钥）
# llm = create_llm("openai", api_key="your-api-key", model_name="gpt-3.5-turbo")

# 或使用Anthropic
# llm = create_llm("anthropic", api_key="your-api-key")

# 2. 初始化动态记忆系统
system = DynamicMemorySystem(llm)

# 3. 存储记忆（自动分类和提取）
memory = system.store("2023年5月15日下午3点，我在星巴克咖啡店遇到了大学同学李明。")
print(f"存储记忆: {memory['type']} (置信度: {memory['confidence']:.2f})")

# 4. 检索记忆（基于语义相似度）
results = system.retrieve("Python", top_k=5)
for mem in results:
    print(f"- {mem['content']}")

# 5. 更新记忆
updated = system.update(memory['id'], "补充信息：我们聊了关于工作的话题")

# 6. 遗忘不重要的记忆
forgotten = system.forget(forget_strategy="low_importance")

# 7. 获取统计信息
stats = system.get_statistics()
print(stats)
```

### 运行示例

```bash
# 运行讯飞星火示例（推荐）
python example_xinghuo_usage.py

# 运行动态记忆系统示例（使用Mock LLM）
python example_dynamic_usage.py

# 直接运行系统核心（查看基本功能）
python memory_system.py
```

### 讯飞星火API配置

1. **获取API密钥**：
   - 访问 [讯飞开放平台](https://www.xfyun.cn/)
   - 创建应用并获取 APPID、APIKey、APISecret

2. **配置API信息**：
   ```python
   llm = create_llm(
       provider="xinghuo",
       appid="75714447",
       api_key="79b6bd157e710cac51c22d357d182870",
       api_secret="NjUzMzNjYTE0MTBiODQ0NWVmZTliZDk5"
   )
   ```

3. **选择API版本**：
   - `v3.1`: 通用版本
   - `v3.5`: 推荐版本（默认）
   - `v4.0`: 最新版本（如果可用）

## 核心功能详解

### 1. 存储（Store）

使用LLM自动理解和分类记忆，而不是硬编码规则：

```python
memory = system.store(
    text="文本内容",
    context={"goal": "当前目标"}  # 可选上下文
)
```

系统会：
- 自动判断记忆类型（情景/语义/程序）
- 提取关键信息（格式自然形成，不强制模板）
- 计算初始重要性分数

### 2. 检索（Retrieve）

基于语义相似度、重要性和访问频率的综合检索：

```python
results = system.retrieve(
    query="查询文本",
    top_k=5,  # 返回前k个结果
    memory_type="semantic"  # 可选：限制记忆类型
)
```

检索会综合考虑：
- **相关性**（0.5权重）：LLM评估的语义相似度
- **重要性**（0.3权重）：Attention Bias计算的重要性
- **访问频率**（0.2权重）：经常访问的记忆优先

### 3. 遗忘（Forget）

智能遗忘策略，保留重要记忆：

```python
# 遗忘低重要性记忆
forgotten = system.forget(forget_strategy="low_importance")

# 遗忘旧的、很少访问的记忆
forgotten = system.forget(forget_strategy="old_unused")

# 遗忘指定记忆
forgotten = system.forget(memory_id=1)
```

### 4. 更新（Update）

动态更新记忆，支持多种更新模式：

```python
# 合并新信息
updated = system.update(memory_id, "新信息", update_mode="merge")

# 替换旧记忆
updated = system.update(memory_id, "新信息", update_mode="replace")

# 精炼和修正
updated = system.update(memory_id, "新信息", update_mode="refine")
```

## 与旧系统的对比

| 特性 | 旧系统 | 新系统（动态记忆） |
|------|--------|-------------------|
| 记忆分类 | 硬编码规则、关键词匹配 | LLM语义理解 |
| 信息提取 | 固定模板、正则表达式 | LLM自然提取 |
| Prompt设计 | 人工编写 | 模型学习优化 |
| 记忆格式 | 强制满足超参数定义 | 自然形成 |
| 记忆管理 | 静态存储 | 动态存储/检索/遗忘/更新 |
| 重要性评估 | 固定规则 | Attention Bias机制 |

## Prompt学习机制

系统内置Prompt学习器，可以根据示例和反馈优化Prompt：

```python
# 优化Prompt（内部自动调用）
examples = [
    {"input": "示例输入", "expected_output": "期望输出"}
]
system.prompt_learner.optimize_prompt("memory_classification", examples)
```

Prompt会随着使用不断演化和改进。

## 配置说明

### 环境变量

```bash
# OpenAI
export OPENAI_API_KEY="your-api-key"

# Anthropic
export ANTHROPIC_API_KEY="your-api-key"
```

### 记忆文件路径

默认记忆存储在 `memory/dynamic_memories.json`，可以通过参数修改：

```python
system = DynamicMemorySystem(llm, memory_file="custom_path.json")
```

## 注意事项

1. **API配置**：
   - 讯飞星火：免费额度，需要配置 APPID、APIKey、APISecret
   - OpenAI/Anthropic：付费服务，需要API密钥
   - Mock模式：测试用，无需API密钥但功能有限

2. **性能**：
   - 每次存储/检索都会调用LLM，可能较慢
   - 讯飞星火使用WebSocket连接，响应速度较快

3. **网络要求**：
   - 讯飞星火API需要能访问 `spark-api.xf-yun.com`
   - 确保网络连接正常

## 扩展开发

### 添加新的LLM后端

在 `llm_interface.py` 中继承 `LLMInterface` 类：

```python
class CustomLLM(LLMInterface):
    def generate(self, prompt, **kwargs):
        # 实现你的LLM调用逻辑
        pass
```

### 自定义Prompt

修改 `PromptLearner` 类中的初始Prompt：

```python
def _initial_classification_prompt(self) -> str:
    return "你的自定义Prompt模板..."
```

## 许可证

本项目遵循MIT许可证。

## 贡献

欢迎提交Issue和Pull Request！
