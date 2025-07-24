from dotenv import load_dotenv
from llama_index.llms.openai import OpenAI
from llama_index.core.agent.workflow import FunctionAgent

load_dotenv()

# Create basic tools
def multiply(a: float, b: float) -> float:
    """Multiply two numbers and returns the product"""
    return a * b


def add(a: float, b: float) -> float:
    """Add two numbers and returns the sum"""
    return a + b

async def main():
    # Initialize the LLM model
    llm = OpenAI(model="gpt-4o-mini")

    # Initialize the agent
    workflow = FunctionAgent(
        tools=[multiply, add],
        llm=llm,
        system_prompt="You are an agent that can perform basic mathematical operations and you must use th provided tools."
    )

    response = await workflow.run(user_msg="What is 20+(2*4)?")
    # the result should be 28
    print(response)


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())