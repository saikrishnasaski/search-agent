from unittest import result

from anthropic.resources.messages import messages
from dotenv import load_dotenv
from langchain import tools
from langchain.tools import tool
from langchain_core.messages import content
from langsmith import traceable
from langchain.messages import SystemMessage, HumanMessage, ToolMessage
from langchain.chat_models import init_chat_model
from messages import AgentMessages
load_dotenv()

MAX_ITERATIONS = 10
MODEL="qwen3:1.7b"
agent_messages = AgentMessages()

# ---- Tools ----- #

@tool
def get_all_available_products() -> list:
    """ Return all available products from catalog"""
    return ['laptop', 'pixel phone', 'samsung tv']

@tool
def get_product_price(product: str) -> float:
    """ Get the price of a product from the following dictionary """
    product_prices = { "laptop": 1299.99, "pixel phone": 999.99, "samsung tv": 1000.99}
    return product_prices.get(product.lower(), 0)

@tool
def apply_discound_based_on_tier(price: float, tier: str) -> float:
    """ Apply the discound based on the tier and return final price """
    tiers = { "gold": 20, "silver": 10, "bronze": 5 }
    discount = tiers.get(tier.lower(), 0)
    final_price = round((price - (price * discount / 100)), 2)
    return final_price


# --- Agent Loop --- #

@traceable(name="Langchain Agent Loop")
def run_agent(question: str):
    print(f"Asking agent about ---- {question}")
    tools=[get_all_available_products, get_product_price, apply_discound_based_on_tier]
    tools_dict={t.name: t for t in tools}

    llm_model=init_chat_model(model=MODEL, temperature=0, model_provider="ollama")
    llm_with_tools=llm_model.bind_tools(tools)

    messages = [
        SystemMessage(content=agent_messages.system_message),
        HumanMessage(content=question)
    ]

    for iteration in range(1, MAX_ITERATIONS + 1):
        print(f"Iteration: {iteration}")

        ai_message = llm_with_tools.invoke(messages)
        tool_calls=ai_message.tool_calls
        print(f"AI Message: {ai_message.content}")
        print(f"\n Tools calls: {tool_calls}")

        if not tool_calls:
            print("No tool calls, ending the loop.")
            print(f"Final Answer: {ai_message.content}")
            return ai_message.content

        # process only one tool-call, only one tool_call per iteration
        tool_call=tool_calls[0]
        tool_name=tool_call.get('name')
        tool_args=tool_call.get('args', {})
        tool_call_id=tool_call.get('id')
        print(f"\nTool name: {tool_name}, Tool call args: {tool_args}, Tool call id: {tool_call_id}")
        tool_call_func=tools_dict.get(tool_name)

        # invoke tool function with arguments
        observation=tool_call_func.invoke(tool_args)
        print(f"\n[Tool Observation]: {observation}")

        messages.append(ai_message)    # adding of history of ai messages to every iteration
        messages.append(
            ToolMessage(content=str(observation), tool_call_id=tool_call_id)
        )   # adding tool result to ai messages

    print("ERROR: Max iterations reached without final answer.")





# --- Main function --- #
if __name__ == "__main__":
    print("Hello from agent!")
    result = run_agent("what is the price of a samsung tv after applying gold tier discount?")
    print(result)
