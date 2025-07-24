from llama_index.llms.openai import OpenAI
from llama_index.tools.tavily_research import TavilyToolSpec
from llama_index.core.agent.workflow import FunctionAgent, AgentStream, AgentInput, AgentOutput, ToolCallResult
from workflows.handler import WorkflowHandler
from dotenv import load_dotenv
import os

load_dotenv()

# Initialize the LLM model
llm = OpenAI(model="gpt-4o-mini")

# Initialize the Tavily tool
# Ensure you have the TAVILY_API_KEY set in your environment variables
# What is Tavily?
# Tavily is a web search tool that allows agents to retrieve information from the web.
tavily_tool = TavilyToolSpec(api_key=os.getenv("TAVILY_API_KEY"))

# Initialize the FunctionAgent with the Tavily tool
workflow = FunctionAgent(
    tools=tavily_tool.to_tool_list(),
    llm=llm,
    system_prompt="You're a helpful assistant that can search the web for information.",
)

async def main():
    # Run the agent with a user message
    # but without the await keyword, since we want to see streaming output
    handler: WorkflowHandler = workflow.run(user_msg="What's the weather like in Cologne?")

    async for event in handler.stream_events():
        if isinstance(event, AgentStream):
            print(event.delta, end="", flush=True)

if __name__ == "__main__":
    import asyncio

    # Run the main function
    asyncio.run(main())



# AgentStream is just one of many events that AgentWorkflow emits as it runs. The others are:
# - AgentInput: the full message object that begins the agent's execution
# - AgentOutput: the response from the agent
# - ToolCall: which tools were called and with what arguments
# - ToolCallResult: the result of a tool call
    """
    async for event in handler.stream_events():
        if isinstance(event, AgentStream):
            print(event.delta, end="", flush=True)
        elif isinstance(event, AgentInput):
            print("Agent input: ", event.input)  # the current input messages
            print("Agent name:", event.current_agent_name)  # the current agent name
        elif isinstance(event, AgentOutput):
            print("Agent output: ", event.response)  # the current full response
            print("Tool calls made: ", event.tool_calls)  # the selected tool calls, if any
            print("Raw LLM response: ", event.raw)  # the raw llm api response
        elif isinstance(event, ToolCallResult):
            print("Tool called: ", event.tool_name)  # the tool name
            print("Arguments to the tool: ", event.tool_kwargs)  # the tool kwargs
            print("Tool output: ", event.tool_output)  # the tool output
    """