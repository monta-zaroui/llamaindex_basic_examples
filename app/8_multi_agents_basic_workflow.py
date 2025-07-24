from llama_index.core.workflow import Context
from llama_index.core.agent.workflow import AgentWorkflow, FunctionAgent, AgentStream
from llama_index.llms.openai import OpenAI
from dotenv import load_dotenv
import re
from tavily import AsyncTavilyClient
import os

from workflows.handler import WorkflowHandler

load_dotenv()

# Initialize the LLM model
llm = OpenAI(model="gpt-4o-mini")


# --- create our tools -------------------------------------------------------------

async def search_web(query: str) -> str:
    """Useful for using the web to answer questions."""
    print("🔍 Searching...", flush=True)
    client = AsyncTavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
    result = str(await client.search(query))
    print("\r✅ Search completed!", flush=True)
    return result


async def record_notes(ctx: Context, notes: str, notes_title: str) -> str:
    """Useful for recording notes on a given topic. Your input should be notes with a title to save the notes under."""
    print("📝 Recording notes...", flush=True)
    async with ctx.store.edit_state() as ctx_state:
        if "research_notes" not in ctx_state["state"]:
            ctx_state["state"]["research_notes"] = {}
        ctx_state["state"]["research_notes"][notes_title] = notes
    print("\r✅ Notes recorded!", flush=True)
    return "Notes recorded."


async def write_report(ctx: Context, report_content: str) -> str:
    """Useful for writing a report on a given topic. Your input should be a markdown-formatted report."""
    print("✏️ Writing report...", flush=True)
    async with ctx.store.edit_state() as ctx_state:
        ctx_state["state"]["report_content"] = report_content
    print("\r✅ Report written!", flush=True)
    return "Report written."


async def review_report(ctx: Context, review: str) -> str:
    """Useful for reviewing a report and providing feedback. Your input should be a review of the report."""
    print("🔍 Reviewing report...", flush=True)
    async with ctx.store.edit_state() as ctx_state:
        ctx_state["state"]["review"] = review
    print("\r✅ Review completed!", flush=True)
    return "Report reviewed."



# --- create our specialist agents ------------------------------------------------

research_agent = FunctionAgent(
    name="ResearchAgent",
    description="Search the web and record notes.",
    system_prompt="You are a researcher… hand off to WriteAgent when ready.",
    llm=llm,
    tools=[search_web, record_notes],
    can_handoff_to=["WriteAgent"],
)

write_agent = FunctionAgent(
    name="WriteAgent",
    description="Writes a markdown report from the notes.",
    system_prompt="You are a writer… ask ReviewAgent for feedback when done.",
    llm=llm,
    tools=[write_report],
    can_handoff_to=["ReviewAgent", "ResearchAgent"],
)

review_agent = FunctionAgent(
    name="ReviewAgent",
    description="Reviews a report and gives feedback.",
    system_prompt="You are a reviewer…",
    llm=llm,
    tools=[review_report],
    can_handoff_to=["WriteAgent"],
)

# --- wire them together ----------------------------------------------------------
agent_workflow = AgentWorkflow(
    agents=[research_agent, write_agent, review_agent],
    root_agent=research_agent.name,
    initial_state={
        "research_notes": {},
        "report_content": "Not written yet.",
        "review": "Review required.",
    },
)

async def main():

    ctx = Context(agent_workflow)

    print("🚀 Starting agent workflow...\n")
    handler: WorkflowHandler = agent_workflow.run(
        user_msg="Write me a report on the history of Agentic AI",
        ctx=ctx,
    )

    async for event in handler.stream_events():
        if isinstance(event, AgentStream):
            print(event.delta, end="", flush=True)

    # Now print the final state after the event loop completes
    print("\n\n--- Workflow run completed! ---")
    ctx_state = await handler.ctx.store.get_state()
    print("Final state:", ctx_state)
    print ("\n-------------------- Research Notes --------------------")
    print(ctx_state["state"]["research_notes"])
    print("\n-------------------- Report Content --------------------")
    print(ctx_state["state"]["report_content"])
    print("\n-------------------- Review --------------------")
    print(ctx_state["state"]["review"])





if __name__ == "__main__":
    import asyncio

    # Run the main function
    asyncio.run(main())