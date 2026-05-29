"""
智能搜索助手 - 基于 LangGraph + Tavily API 的真实搜索系统
1. 理解用户需求
2. 使用Tavily API真实搜索信息  
3. 生成基于搜索结果的回答
"""

import asyncio
from typing import TypedDict, Annotated
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field
from langgraph.checkpoint.memory import InMemorySaver
import os
from dotenv import load_dotenv
from tavily import TavilyClient

# 加载环境变量
load_dotenv()

class CodeState(TypedDict):
    messages: Annotated[list, add_messages]
    target_product: str       # 用户最终想得到什么，例如 Python 脚本、函数、类、API、网页、配置文件等
    user_query: str          # 用户原始需求
    understanding: str       # 对用户需求的理解
    assumptions: list[str]   # 模型做出的合理假设
    dependencies: list[str]  # 需要安装的依赖
    python_code: str         # 生成的 Python 代码
    run_command: str         # 运行命令
    notes: str               # 补充说明
    error: str               # 错误信息
    final_answer: str        # 最终回答
    step: str                # 当前步骤

class Understanding(BaseModel):
    understanding: str = Field(description="用一句话总结用户想要实现什么")
    target_product: str = Field(description="用户最终想得到什么，例如 Python 脚本、函数、类、API、网页、配置文件等")
    technical_constraints: str = Field(description="技术约束，例如语言、框架、库、运行环境等，没有则写“未指定”")
    input_info: str = Field(description="代码需要接收什么输入，没有则写“未指定”")
    output_info: str = Field(description="代码需要产生什么输出，没有则写“未指定”")
    function_points: list[str] = Field(default_factory=list, description="列出，核心功能要点")
    assumptions: list[str] = Field(default_factory=list, description="需求不完整时可以采用的默认假设，没有则写“无")
    clarifying_questions: list[str] = Field(default_factory=list, description="影响实现的关键问题，没有则写“无”")

class PythonCodeOutput(BaseModel):
    understanding: str = Field(description="用一句话总结用户想要实现什么")
    assumptions: list[str] = Field(default_factory=list, description="合理假设")
    dependencies: list[str] = Field(default_factory=list, description="需要安装的第三方库")
    python_code: str = Field(description="完整、可运行的 Python 代码")
    run_command: str = Field(default="python main.py", description="运行代码的命令")
    notes: str = Field(default="", description="补充说明")

# 初始化模型和Tavily客户端
llm = ChatOpenAI(
    model=os.getenv("LLM_MODEL_ID", "gpt-4o-mini"),
    api_key=os.getenv("LLM_API_KEY"),
    base_url=os.getenv("LLM_BASE_URL", "https://api.openai.com/v1"),
    temperature=0.7
)


def understand_query_node(state: CodeState) -> CodeState:
    """步骤1：理解用户查询并生成搜索关键词"""
    
    # 获取最新的用户消息
    user_message = ""
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage):
            user_message = msg.content
            break
    
    understand_prompt = f"""你是一个需求分析助手。请分析用户的开发需求："{user_message}"

  你的任务是把用户的自然语言需求整理成适合后续代码生成的需求说明。

  请输出：
  需求理解：[一句话说明用户想实现的功能]
  目标产物：[用户最终想得到什么，例如 Python 脚本、函数、类、API、网页、配置文件等]
  技术约束：[语言、框架、库、运行环境等，没有则写“未指定”]
  输入信息：[代码需要接收什么输入，没有则写“未指定”]
  输出信息：[代码需要产生什么输出，没有则写“未指定”]
  功能要点：[列出核心功能点]
  合理假设：[需求不完整时可以采用的默认假设，没有则写“无”]
  待澄清问题：[影响实现的关键问题，没有则写“无”]

  要求：
  - 只做需求分析，不要生成代码
  - 不要输出搜索关键词
  - 不要使用 Markdown 代码块
  - 内容要简洁、准确、便于后续代码生成模型使用
  """
    structured_llm = llm.with_structured_output(
      Understanding,
      method="function_calling"
  )

    response: Understanding = structured_llm.invoke([
      SystemMessage(content=understand_prompt)
    ])

    response_text = (
      f"需求理解：{response.understanding}\n"
      f"目标产物：{response.target_product}\n"
      f"技术约束：{response.technical_constraints}\n"
      f"输入信息：{response.input_info}\n"
      f"输出信息：{response.output_info}\n"
      f"功能要点：{', '.join(response.function_points) if response.function_points else '无'}\n"
      f"合理假设：{', '.join(response.assumptions) if response.assumptions else '无'}\n"
      f"待澄清问题：{', '.join(response.clarifying_questions) if response.clarifying_questions else '无'}"
    )

    print(f"🔍 理解阶段输出:\n{response_text}\n")

    return {
      "understanding": response.understanding,
      "user_query": user_message,
      "assumptions": response.assumptions,
      "step": "understood",
      "target_product": response.target_product, 
      "messages": [AIMessage(content=f"我理解您的需求：\n{response_text}")]
       
        }



def generate_answer_node(state: CodeState) -> CodeState:
    """步骤3：基于搜索结果生成最终答案"""
    
    # 检查是否有搜索结果
    if state["step"] == "search_failed":
        # 如果搜索失败，基于LLM知识回答
        fallback_prompt = f"""搜索API暂时不可用，请基于您的知识回答用户的问题：

用户问题：{state['user_query']}

请提供一个有用的回答，并说明这是基于已有知识的回答。"""
        
        response = llm.invoke([SystemMessage(content=fallback_prompt)])
        
        return {
            "final_answer": response.content,
            "step": "completed",
            "messages": [AIMessage(content=response.content)]
        }
    
    # 基于搜索结果生成答案
    answer_prompt = f"""基于以下搜索结果为用户提供完整、准确的答案：

用户问题：{state['user_query']}

搜索结果：
{state['search_results']}

请要求：
1. 综合搜索结果，提供准确、有用的回答
2. 如果是技术问题，提供具体的解决方案或代码
3. 引用重要信息的来源
4. 回答要结构清晰、易于理解
5. 如果搜索结果不够完整，请说明并提供补充建议"""

    response = llm.invoke([SystemMessage(content=answer_prompt)])
    
    return {
        "final_answer": response.content,
        "step": "completed",
        "messages": [AIMessage(content=response.content)]
    }

# 构建搜索工作流
def create_search_assistant():
    workflow = StateGraph(SearchState)
    
    # 添加三个节点
    workflow.add_node("understand", understand_query_node)
    workflow.add_node("search", tavily_search_node)
    workflow.add_node("answer", generate_answer_node)
    
    # 设置线性流程
    workflow.add_edge(START, "understand")
    workflow.add_edge("understand", "search")
    workflow.add_edge("search", "answer")
    workflow.add_edge("answer", END)
    
    # 编译图
    memory = InMemorySaver()
    app = workflow.compile(checkpointer=memory)
    
    return app

async def main():
    """主函数：运行智能搜索助手"""
    
    # 检查API密钥
    if not os.getenv("TAVILY_API_KEY"):
        print("❌ 错误：请在.env文件中配置TAVILY_API_KEY")
        return
    
    app = create_search_assistant()
    
    print("🔍 智能搜索助手启动！")
    print("我会使用Tavily API为您搜索最新、最准确的信息")
    print("支持各种问题：新闻、技术、知识问答等")
    print("(输入 'quit' 退出)\n")
    
    session_count = 0
    
    while True:
        user_input = input("🤔 您想了解什么: ").strip()
        
        if user_input.lower() in ['quit', 'q', '退出', 'exit']:
            print("感谢使用！再见！👋")
            break
        
        if not user_input:
            continue
        
        session_count += 1
        config = {"configurable": {"thread_id": f"search-session-{session_count}"}}
        
        # 初始状态
        initial_state = {
            "messages": [HumanMessage(content=user_input)],
            "user_query": "",
            "search_query": "",
            "search_results": "",
            "final_answer": "",
            "step": "start"
        }
        
        try:
            print("\n" + "="*60)
            
            # 执行工作流
            async for output in app.astream(initial_state, config=config):
                for node_name, node_output in output.items():
                    if "messages" in node_output and node_output["messages"]:
                        latest_message = node_output["messages"][-1]
                        if isinstance(latest_message, AIMessage):
                            if node_name == "understand":
                                print(f"🧠 理解阶段: {latest_message.content}")
                            elif node_name == "search":
                                print(f"🔍 搜索阶段: {latest_message.content}")
                            elif node_name == "answer":
                                print(f"\n💡 最终回答:\n{latest_message.content}")
            
            print("\n" + "="*60 + "\n")
        
        except Exception as e:
            print(f"❌ 发生错误: {e}")
            print("请重新输入您的问题。\n")

if __name__ == "__main__":
    # asyncio.run(main())
    def test_understand_query_node():
        state = {
            "messages": [HumanMessage(content="写一个 Python 函数，判断字符串是否是回文")],
            "user_query": "",
            "understanding": "",
            "assumptions": [],
            "dependencies": [],
            "python_code": "",
            "run_command": "",
            "notes": "",
            "error": "",
            "final_answer": "",
            "step": "start",
        }

        result = understand_query_node(state)

        print("返回结果：", result)

        assert result["step"] == "understood"
        assert result["user_query"] == "写一个 Python 函数，判断字符串是否是回文"
        assert "understanding" in result
        assert result["understanding"] != ""
        assert "messages" in result
    test_understand_query_node()