from unittest import result

from dotenv import load_dotenv
from langchain import tools
from langchain.tools import tool
from langsmith import traceable
load_dotenv()

MAX_ITERATIONS = 10

# ---- Tools ----- #

@tool
def get_product_price(product: str) -> float:
    """ Get the price of a product from the following dictionary """
    product_prices = { "laptop": 1299.99, "pixel phone": 999.99, "samsung tv": 1000.99}
    return product_prices[product, 0]

@tool
def apply_discound_based_on_tier(price: float, tier: str) -> float:
    """ Apply the discound based on the tier and return final price """
    tiers = { "gold": 20, "silver": 10, "bronze": 5 }
    discount = tiers[tier, 0]
    final_price = round((price - (price * discount / 100)), 2)
    return final_price


# --- Agent Loop --- #

@traceable(name="Langchain Agent Loop")
def run_agent(question: str):
    pass




# --- Main function --- #
if __name__ == "__main__":
    print("Hello from agent!")
    result = run_agent("what is the price of a pixel phone after applying gold tier discount?")
    print(result)
