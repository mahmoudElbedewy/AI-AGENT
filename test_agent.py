
# from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain_core.tools import tool
# from langgraph.prebuilt import create_react_agent
# from langchain_community.tools import DuckDuckGoSearchRun

# searchMethod = DuckDuckGoSearchRun()

# @tool
# def calculator(expression: str) -> str:
#     """تستخدم هذه الأداة كآلة حاسبة. مرر لها المعادلة الرياضية وسوف تعيد لك الناتج."""
#     try:
#         return str(eval(expression))
#     except Exception as e:
#         return f"خطأ في الحساب: {e}"

# llm = ChatGoogleGenerativeAI(
#     model="gemini-2.5-flash-lite",
#     google_api_key='AIzaSyDMnAK_PDsEV6HeczdPXHlaOZvcAc1Oe3w',
#     temperature = 0
# )

# tools = [calculator , searchMethod]

# agent_executor = create_react_agent(
#     llm, 
#     tools,
#     prompt="أنت مساعد ذكي وباحث. استخدم أدوات البحث للحصول على معلومات حديثة، واستخدم الآلة الحاسبة لأي عمليات رياضية."
# )

# question = input('اى سؤالك')

# response = agent_executor.invoke({"messages": [("user", question)]})
# print("\nالرد النهائي للعميل:")

# print(response['messages'][-1].content)