import re
from llama_index.core.agent.workflow import FunctionAgent, AgentStream, AgentOutput, ToolCallResult, ToolCall
from llama_index.core.workflow import Context
from llama_index.llms.openai import OpenAI
from dotenv import load_dotenv
from tavily import AsyncTavilyClient
import os
load_dotenv()


# Initialize the LLM models
sub_agent_llm = OpenAI(model="gpt-4.1-mini")
orchestrator_llm = OpenAI(model="o3-mini")

# ------------------ create our tools ----------------------------------------
async def search_web(query: str) -> str:
    """Useful for using the web to answer questions."""
    client = AsyncTavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
    return str(await client.search(query))


# ------------------ create our specialist agents -----------------------------

research_agent = FunctionAgent(
    system_prompt=(
        "You are the ResearchAgent that can search the web for information on a given topic and record notes on the topic. "
        "You should output notes on the topic in a structured format."
    ),
    llm=sub_agent_llm,
    tools=[search_web],
)

write_agent = FunctionAgent(
    system_prompt=(
        "You are the WriteAgent that can write a report on a given topic. "
        "Your report should be in a markdown format. The content should be grounded in the research notes. "
        "Return your markdown report surrounded by ... tags."
    ),
    llm=sub_agent_llm,
    tools=[],
)

review_agent = FunctionAgent(
    system_prompt=(
        "You are the ReviewAgent that can review the write report and provide feedback. "
        "Your review should either approve the current report or request changes to be implemented."
    ),
    llm=sub_agent_llm,
    tools=[],
)


# ------------------ convert specialist agents into tools -----------------------------
async def call_research_agent(ctx: Context, prompt: str) -> str:
    """Useful for recording research notes based on a specific prompt."""
    result = await research_agent.run(
        user_msg=f"Write some notes about the following: {prompt}"
    )

    async with ctx.store.edit_state() as ctx_state:
        ctx_state["state"]["research_notes"].append(str(result))

    return str(result)


async def call_write_agent(ctx: Context) -> str:
    """Useful for writing a report based on the research notes or revising the report based on feedback."""
    async with ctx.store.edit_state() as ctx_state:
        notes = ctx_state["state"].get("research_notes", None)
        if not notes:
            return "No research notes to write from."

        user_msg = f"Write a markdown report from the following notes. Be sure to output the report in the following format: <report>...</report>:\n\n"

        # Add the feedback to the user message if it exists
        feedback = ctx_state["state"].get("review", None)
        if feedback:
            user_msg += f"<feedback>{feedback}</feedback>\n\n"

        # Add the research notes to the user message
        notes = "\n\n".join(notes)
        user_msg += f"<research_notes>{notes}</research_notes>\n\n"

        # Run the write agent
        result = await write_agent.run(user_msg=user_msg)
        report = re.search(
            r"<report>(.*)</report>", str(result), re.DOTALL
        ).group(1)
        ctx_state["state"]["report_content"] = str(report)

    return str(report)


async def call_review_agent(ctx: Context) -> str:
    """Useful for reviewing the report and providing feedback."""
    async with ctx.store.edit_state() as ctx_state:
        report = ctx_state["state"].get("report_content", None)
        if not report:
            return "No report content to review."

        result = await review_agent.run(
            user_msg=f"Review the following report: {report}"
        )
        ctx_state["state"]["review"] = result

    return result


# ------------------ Top-level orchestrator agent ----------------------------------------
orchestrator = FunctionAgent(
    system_prompt=(
        "You are an expert in the field of report writing. "
        "You are given a user request and a list of tools that can help with the request. "
        "You are to orchestrate the tools to research, write, and review a report on the given topic. "
        "Once the review is positive, you should notify the user that the report is ready to be accessed."
    ),
    llm=orchestrator_llm,
    tools=[
        call_research_agent,
        call_write_agent,
        call_review_agent,
    ],
    initial_state={
        "research_notes": [],
        "report_content": None,
        "review": None,
    },
)

# ------------------ Run the orchestrator agent func ----------------------------------------
async def run_orchestrator(ctx: Context, user_msg: str):
    handler = orchestrator.run(
        user_msg=user_msg,
        ctx=ctx,
    )

    async for event in handler.stream_events():
        if isinstance(event, AgentStream):
            if event.delta:
                print(event.delta, end="", flush=True)
        # elif isinstance(event, AgentInput):
        #     print("📥 Input:", event.input)
        elif isinstance(event, AgentOutput):
            # Skip printing the output since we are streaming above
            # if event.response.content:
            #     print("📤 Output:", event.response.content)
            if event.tool_calls:
                print(
                    "🛠️  Planning to use tools:",
                    [call.tool_name for call in event.tool_calls],
                )
        elif isinstance(event, ToolCallResult):
            print(f"🔧 Tool Result ({event.tool_name}):")
            print(f"  Arguments: {event.tool_kwargs}")
            print(f"  Output: {event.tool_output}")
        elif isinstance(event, ToolCall):
            print(f"🔨 Calling Tool: {event.tool_name}")
            print(f"  With arguments: {event.tool_kwargs}")
async def main():
    # Create a context for the orchestrator to hold history/state
    ctx = Context(orchestrator)

    await run_orchestrator(
        ctx=ctx,
        user_msg=(
            "Write me a report on the history of the internet. "
            "Briefly describe the history of the internet, including the development of the internet, the development of the web, "
            "and the development of the internet in the 21st century."
        ),
    )

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())