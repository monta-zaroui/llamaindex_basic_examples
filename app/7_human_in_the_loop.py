from llama_index.core.agent.workflow import FunctionAgent
from llama_index.core.workflow import InputRequiredEvent, HumanResponseEvent, Context
from llama_index.llms.openai import OpenAI
from dotenv import load_dotenv

load_dotenv()



async def dangerous_task(ctx: Context) -> str:
    """A dangerous task that requires human confirmation."""

    # emit a waiter event (InputRequiredEvent here)
    # and wait until we see a HumanResponseEvent
    question = "Are you sure you want to proceed? "
    response = await ctx.wait_for_event(
        HumanResponseEvent,
        waiter_id=question,
        waiter_event=InputRequiredEvent(
            prefix=question,
        ),
    )

    # act on the input from the event
    if response.response.strip().lower() == "yes":
        return "Dangerous task completed successfully."
    else:
        return "Dangerous task aborted."

# Initialize the LLM model
llm = OpenAI(model="gpt-4o-mini")

# Initialize the FunctionAgent with the dangerous task
workflow = FunctionAgent(
    tools=[dangerous_task],
    llm=llm,
    system_prompt="You are a helpful assistant that can perform dangerous tasks.",
)

async def main():
    handler = workflow.run(user_msg="I want to proceed with the dangerous task.")

    async for event in handler.stream_events():
        if isinstance(event, InputRequiredEvent):
            # capture keyboard input
            response = input(event.prefix)
            # send our response back
            handler.ctx.send_event(
                HumanResponseEvent(
                    response=response,
                )
        )

    response = await handler
    print(response)

if __name__ == "__main__":
    import asyncio

    # Run the main function
    asyncio.run(main())