"""
AutoGen 软件开发团队协作案例
"""

import os
import asyncio
import subprocess
import sys
import tempfile
from typing import List, Dict, Any
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 先测试一个版本，使用 OpenAI 客户端
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_agentchat.agents import AssistantAgent, UserProxyAgent
from autogen_agentchat.teams import RoundRobinGroupChat,SelectorGroupChat
from autogen_agentchat.conditions import TextMentionTermination
from autogen_agentchat.ui import Console
selector_prompt = """
你是软件开发团队的对话调度器。

可选角色：
{participants}

角色说明：
{roles}

对话历史：
{history}

请根据语义选择下一位发言者。

调度规则：
1. 初始需求应先交给 ProductManager 分析。
2. ProductManager 完成需求分析后，交给 Engineer 实现。
3. Engineer 完成初次实现后，交给 CodeReviewer 审查。
4. CodeReviewer 如果发现代码问题，交回 Engineer。
5. CodeReviewer 如果认为代码可以测试，交给 QualityAssurance。
6. QualityAssurance 使用工具测试代码；如果测试失败，交回 Engineer；如果测试通过，交给 UserProxy。
7. UserProxy 如果只是测试失败或发现 bug，交给 Engineer 修改。
8. UserProxy 如果修改了需求，必须交给 Engineer 基于当前代码重新修改。
9. Engineer 根据 UserProxy 的需求变更修改代码后，必须交给 ProductManager 重新审核需求一致性。
10. ProductManager 重新审核通过后，再交给 CodeReviewer。
11. UserProxy 验收通过并回复 TERMINATE 时，对话结束。

只返回一个角色名，不要解释。
"""
def create_openai_model_client():
    """创建 OpenAI 模型客户端用于测试"""
    return OpenAIChatCompletionClient(
        model=os.getenv("LLM_MODEL_ID", "gpt-4o"),
        api_key=os.getenv("LLM_API_KEY"),
        base_url=os.getenv("LLM_BASE_URL", "https://api.openai.com/v1"),
        model_info={
            "function_calling": True,
            "max_tokens": 4096,
            "context_length": 32768,
            "vision": False,
            "json_output": True,
            "family": "deepseek",
            "structured_output": True,
        }
    )

def test_python_code(code: str, run_script: bool = True, timeout: int = 10) -> Dict[str, Any]:
    """测试一段 Python 代码是否可编译、可运行。

    参数：
    - code: 需要测试的完整 Python 代码。
    - run_script: 是否在编译通过后运行脚本。
    - timeout: 运行超时时间，单位秒，最大 30 秒。

    返回：
    - ok: 是否通过测试。
    - compile_returncode: 编译退出码。
    - run_returncode: 运行退出码，未运行时为 None。
    - stdout: 运行标准输出。
    - stderr: 编译或运行错误输出。
    """
    timeout = max(1, min(timeout, 30))

    with tempfile.TemporaryDirectory(prefix="autogen_python_test_") as temp_dir:
        file_path = os.path.join(temp_dir, "candidate.py")
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(code)

        compile_result = subprocess.run(
            [sys.executable, "-m", "py_compile", file_path],
            cwd=temp_dir,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if compile_result.returncode != 0:
            return {
                "ok": False,
                "compile_returncode": compile_result.returncode,
                "run_returncode": None,
                "stdout": compile_result.stdout,
                "stderr": compile_result.stderr,
            }

        if not run_script:
            return {
                "ok": True,
                "compile_returncode": compile_result.returncode,
                "run_returncode": None,
                "stdout": compile_result.stdout,
                "stderr": compile_result.stderr,
            }

        run_result = subprocess.run(
            [sys.executable, file_path],
            cwd=temp_dir,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return {
            "ok": run_result.returncode == 0,
            "compile_returncode": compile_result.returncode,
            "run_returncode": run_result.returncode,
            "stdout": run_result.stdout,
            "stderr": run_result.stderr,
        }

def create_product_manager(model_client):
    """创建产品经理智能体"""
    system_message = """你是软件开发团队中的产品经理，负责需求分析、需求变更评估、验收标准定义和产品一致性审核。

你的核心职责：
1. 理解用户原始需求，拆解功能模块、边界条件和验收标准。
2. 当 UserProxy 提出新需求、修改需求或验收反馈时，判断这些变化是否清晰、合理、可实现。
3. 审核 Engineer 提交的实现是否满足当前最新需求。
4. 如果实现与需求不一致，明确指出缺口，并要求 Engineer 修改。
5. 如果实现满足需求，说明可以进入 CodeReviewer 审查或 UserProxy 验收。

工作规则：
1. 始终基于“最新需求”进行判断。
2. 如果 UserProxy 修改了需求，你必须重新确认需求范围、影响范围和验收标准。
3. 不要编写完整代码，代码实现由 Engineer 完成。
4. 不要只给笼统意见，必须给出可执行的修改点或明确审核结论。
5. 如果需求不清楚，要求 UserProxy 澄清。
6. 如果需求清楚但代码未满足，要求 Engineer 修改。
7. 如果需求和实现一致，允许进入 CodeReviewer 审查。

请按以下格式输出：
1. 当前需求理解
2. 验收标准
3. 实现一致性判断
4. 风险或缺口
5. 下一步建议

下一步建议只能从以下动作中选择：
- 交给 Engineer 修改
- 交给 CodeReviewer 审查
- 交给 UserProxy 澄清"""

    return AssistantAgent(
        name="ProductManager",
        model_client=model_client,
        system_message=system_message,
    )

def create_engineer(model_client):
    """创建软件工程师智能体"""
    system_message = """你是软件开发团队中的资深软件工程师，负责根据当前最新需求实现、修改和交付代码。

你的核心职责：
1. 根据 ProductManager 的需求分析编写完整、可运行的代码。
2. 根据 CodeReviewer 的审查意见修复代码问题。
3. 根据 UserProxy 的测试反馈或需求变更更新代码。
4. 如果 UserProxy 修改了需求，你必须基于新需求重新调整实现，并说明修改点。
5. 修改完成后，需要把实现交给 ProductManager 重新审核需求一致性，而不是直接跳到最终验收。

工作规则：
1. 必须保留并理解当前最新需求。
2. 如果收到的是需求变更，不要只修 bug，要重新评估代码是否覆盖新需求。
3. 如果代码来自上一轮，你需要在其基础上修改，不要无视已有实现重新写无关版本。
4. 不要替 ProductManager 做需求审核。
5. 不要替 CodeReviewer 做正式代码审查。
6. 如果需求存在矛盾，先请求 ProductManager 或 UserProxy 澄清。

每次提交代码时，请按以下格式输出：
1. 需求依据
2. 修改/实现摘要
3. 完整代码
4. 运行方式
5. 自检结果
6. 下一步建议

下一步建议通常是：
- 交给 ProductManager 重新审核
- 交给 CodeReviewer 审查
- 请求 UserProxy 澄清需求"""

    return AssistantAgent(
        name="Engineer",
        model_client=model_client,
        system_message=system_message,
    )

def create_code_reviewer(model_client):
    """创建代码审查员智能体"""
    system_message = """你是软件开发团队中的代码审查员，负责检查 Engineer 提交的代码质量、正确性、安全性、可维护性和需求覆盖情况。

你的核心职责：
1. 审查代码是否完整、清晰、可运行。
2. 检查代码是否符合当前最新需求和 ProductManager 定义的验收标准。
3. 检查错误处理、边界情况、依赖使用、性能和安全风险。
4. 如果发现阻塞问题，给出具体修改建议，并要求 Engineer 修改。
5. 如果代码质量通过，交给 UserProxy 测试或验收。
6. 如果发现需求和实现之间存在产品层面的冲突，建议交给 ProductManager 重新审核。

工作规则：
1. 不要重写完整代码，除非只是给出局部示例。
2. 审查意见必须具体到问题、影响和建议。
3. 如果问题会导致功能失败，必须明确要求 Engineer 修改。
4. 如果只是轻微优化，可以说明不阻塞验收。
5. 如果 UserProxy 修改过需求，你必须确认当前代码是否对应最新需求。
6. 如果无法判断需求是否被满足，应建议交给 ProductManager 重新审核。

请按以下格式输出：
1. 审查结论
2. 阻塞问题
3. 非阻塞建议
4. 需求覆盖判断
5. 下一步建议

下一步建议只能从以下动作中选择：
- 交给 Engineer 修改
- 交给 ProductManager 重新审核
- 交给 UserProxy 测试"""

    return AssistantAgent(
        name="CodeReviewer",
        model_client=model_client,
        system_message=system_message,
    )

def create_user_proxy():
    """创建用户代理智能体"""
    return UserProxyAgent(
        name="UserProxy",
        description="""你是用户代理，负责代表真实用户提出需求、测试结果、验收意见和需求变更。

你的核心职责：
1. 提供初始用户需求。
2. 阅读 Engineer 提交的代码和说明。
3. 根据 ProductManager 的验收标准进行测试和验收。
4. 如果功能不符合预期，说明具体失败点。
5. 如果用户修改了需求，必须明确说明这是“需求变更”，并描述新增需求、删除的旧需求、保留的旧需求。
6. 当你修改需求后，需要让 Engineer 基于当前代码重新修改，然后交给 ProductManager 重新审核。
7. 如果测试通过并且没有新需求，回复 TERMINATE。

工作规则：
1. 不要直接修改代码。
2. 不要只说“不满意”，必须说明哪里不符合预期。
3. 需求变更必须清晰描述，不要混在普通测试反馈里。
4. 新增功能、删除功能、改变交互、改变技术要求，都视为需求变更。
5. 如果只是运行错误或 bug，视为测试反馈。
6. 如果需求变更后，下一步应交给 Engineer 修改代码。
7. Engineer 修改后，必须由 ProductManager 重新审核需求一致性。

请按以下格式输出：
1. 测试/验收结果
2. 是否存在需求变更
3. 需求变更详情
4. 问题或反馈
5. 下一步建议

如果验收通过：
回复 TERMINATE。

如果修改了需求：
下一步建议应为：交给 Engineer 根据新需求修改代码，并在修改后交给 ProductManager 重新审核。""",
    )
def create_quality_assurance(model_client):
    """创建质量保证智能体"""
    system_message = """你是软件开发团队中的质量保证专家，负责确保最终产品符合所有既定的质量标准和用户期望。

你的核心职责：
1. 根据 ProductManager 定义的验收标准进行测试。
2. 识别并记录任何不符合要求的功能或性能问题。
3. 与 Engineer 和 CodeReviewer 协作，确保问题得到解决。
4. 在最终验收阶段提供专业的质量评估。

工作规则：
1. 始终基于“最新需求”进行测试。
2. 当需要验证 Python 代码时，必须优先使用 test_python_code 工具进行编译和运行测试。
3. 如果发现质量问题，必须提供具体的重现步骤、工具输出和影响分析。
4. 不要直接修改代码，而是通过沟通推动问题解决。
5. 如果问题严重，必须要求 Engineer 修改。
6. 如果问题轻微，可以提出改进建议。
7. 调用 test_python_code 工具后，必须基于工具返回值输出最终测试结论，不能只停留在工具调用结果。
8. 最终结论必须明确说明 ok、compile_returncode、run_returncode、stdout、stderr 的关键信息。

请按以下格式输出：
1. 测试结果
2. 发现的问题
3. 问题影响分析
4. 改进建议
5. 下一步建议

下一步建议只能从以下动作中选择：
- 交给 Engineer 修改
- 交给 CodeReviewer 审查
- 交给 ProductManager 重新审核"""

    return AssistantAgent(
        name="QualityAssurance",
        model_client=model_client,
        system_message=system_message,
        tools=[test_python_code],
        reflect_on_tool_use=True,
    )
def candidate_func(messages):
    if not messages:
        return ["ProductManager"]

    last = messages[-1]
    source = getattr(last, "source", "")

    if source == "ProductManager":
        return ["Engineer"]

    if source == "Engineer":
        return ["CodeReviewer","ProductManager"]

    if source == "CodeReviewer":
        # 让模型判断是回工程师修改，还是进入质量测试
        return ["Engineer", "QualityAssurance"]

    if source == "QualityAssurance":
        # 让模型判断是回工程师修复，还是进入用户验收
        return ["Engineer", "UserProxy"]

    if source == "UserProxy":
        # 让模型判断是否需要工程师修复
        return ["Engineer"]

    return ["ProductManager", "Engineer", "CodeReviewer", "QualityAssurance", "UserProxy"]
async def run_software_development_team():
    """运行软件开发团队协作"""
    
    print("🔧 正在初始化模型客户端...")
    
    # 先使用标准的 OpenAI 客户端测试
    model_client = create_openai_model_client()
    
    print("👥 正在创建智能体团队...")
    
    # 创建智能体团队
    product_manager = create_product_manager(model_client)
    engineer = create_engineer(model_client)
    code_reviewer = create_code_reviewer(model_client)
    quality_assurance = create_quality_assurance(model_client)
    user_proxy = create_user_proxy()
    
    # 添加终止条件
    termination = TextMentionTermination("TERMINATE")
    
    # 创建团队聊天
    team_chat = SelectorGroupChat(
        participants=[
            product_manager,
            engineer,
            code_reviewer,
            quality_assurance,
            user_proxy,
        ],
        model_client=model_client,
        selector_prompt=selector_prompt,
        candidate_func=candidate_func,
        termination_condition=termination,
        max_turns=20,
    )
    
    # 定义开发任务
    task = """我们需要开发一个比特币价格显示应用，具体要求如下：

核心功能：
- 实时显示比特币当前价格（USD）
- 显示24小时价格变化趋势（涨跌幅和涨跌额）
- 提供价格刷新功能

技术要求：
- 使用 Streamlit 框架创建 Web 应用
- 界面简洁美观，用户友好
- 添加适当的错误处理和加载状态

请团队协作完成这个任务，从需求分析到最终实现。"""
    
    # 执行团队协作
    print("🚀 启动 AutoGen 软件开发团队协作...")
    print("=" * 60)
    
    # 使用 Console 来显示对话过程
    result = await Console(team_chat.run_stream(task=task))
    
    print("\n" + "=" * 60)
    print("✅ 团队协作完成！")
    
    return result

# 主程序入口
if __name__ == "__main__":
    try:
        # 运行异步协作流程
        result = asyncio.run(run_software_development_team())
        
        print(f"\n📋 协作结果摘要：")
        print(f"- 参与智能体数量：4个")
        print(f"- 任务完成状态：{'成功' if result else '需要进一步处理'}")
        
    except ValueError as e:
        print(f"❌ 配置错误：{e}")
        print("请检查 .env 文件中的配置是否正确")
    except Exception as e:
        print(f"❌ 运行错误：{e}")
        import traceback
        traceback.print_exc()
