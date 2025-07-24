from dotenv import load_dotenv
from llama_index.llms.openai import OpenAI
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.core.workflow import JsonSerializer, Context

load_dotenv()

# The Context is serializable, so it can be saved to a database, file, etc. and loaded back in later.
# The JsonSerializer is a simple serializer that uses json.dumps and json.loads to serialize and deserialize the context.
# The JsonPickleSerializer is a serializer that uses pickle to serialize and deserialize the context.
# If you have objects in your context that are not serializable, you can use this serializer.
async def main():
    # Initialize the LLM
    llm = OpenAI(model="gpt-4o-mini")

    # Initialize the agent
    workflow = FunctionAgent(lm=llm)


    # Initialize the workflow context
    ctx = Context(workflow)

    # Run the workflow with a user message
    response = await workflow.run(user_msg="Hi, my name is Laurie!", ctx=ctx)
    print(response)

    # Run the workflow again with a different user message to check the maintained state
    response2 = await workflow.run(user_msg="What's my name?", ctx=ctx)
    print(response2)

    # convert our Context to a dictionary object
    ctx_dict = ctx.to_dict(serializer=JsonSerializer())

    # create a new Context from the dictionary
    restored_ctx = Context.from_dict(
        workflow, ctx_dict, serializer=JsonSerializer()
    )

    # run the workflow again with the restored context
    response3 = await workflow.run(user_msg="What's my name?", ctx=restored_ctx)
    print(response3)

if __name__ == "__main__":
    import asyncio

    asyncio.run(main())