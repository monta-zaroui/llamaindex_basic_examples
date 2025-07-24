from dotenv import load_dotenv
from llama_index.llms.openai import OpenAI
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.tools.yahoo_finance import YahooFinanceToolSpec

load_dotenv()

# Create basic tools
def multiply(a: float, b: float) -> float:
    """Multiply two numbers and returns the product"""
    return a * b


def add(a: float, b: float) -> float:
    """Add two numbers and returns the sum"""
    return a + b

async def main():
    # Initialize the LLM
    llm = OpenAI(model="gpt-4o-mini")

    # Load Yahoo Finance tools
    finance_tools = YahooFinanceToolSpec().to_tool_list()

    # Extend the tools with basic operations
    finance_tools.extend([multiply, add])

    # Initialize the agent
    workflow = FunctionAgent(
        name="Stock Agent",
        description="Useful for performing financial operations.",
        llm=llm,
        tools=finance_tools,
        system_prompt="You are a helpful assistant.",
    )

    response = await workflow.run(user_msg="What's the current stock price of Deutsche Telekom?")

    print(response)


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())