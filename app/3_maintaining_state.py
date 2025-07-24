from dotenv import load_dotenv
from llama_index.llms.openai import OpenAI
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.core.workflow import Context

load_dotenv()

# To maintain the state, we need to keep track of the previous state.
# In LlamaIndex, Workflows have a Context class that can be used to maintain state within and between runs.
# Since the AgentWorkflow is just a pre-built Workflow, we can also use it now.
async def main():
    # Initialize the LLM
    llm = OpenAI(model="gpt-4o-mini")

    workflow = FunctionAgent(name= 'Agent',llm=llm)

    # Initialize the workflow context
    # To maintain state between runs, we'll create a new Context called ctx.
    # We pass in our workflow to properly configure this Context object for the workflow that will use it.
    ctx = Context(workflow)

    # Run the workflow with a user message
    response = await workflow.run(user_msg="Hi, my name is Laurie!", ctx=ctx)
    print(response)

    # Run the workflow again with a different user message to check the maintained state
    response2 = await workflow.run(user_msg="What's my name?", ctx=ctx)
    print(response2)

if __name__ == "__main__":
    import asyncio

    asyncio.run(main())