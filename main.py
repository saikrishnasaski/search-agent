from typing import List
from pydantic import BaseModel, Field
import tavily
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain.tools import tool
from langchain.messages import HumanMessage
from langchain_ollama import ChatOllama
from langchain.agents import create_agent
import keyring
from langchain_tavily import TavilySearch

load_dotenv()

class Source(BaseModel):
    """Schema for a source used by the agent"""
    url: str = Field(description="The URL of the source")

class AgentResponse(BaseModel):
    """Schema for a agent response used by the agent"""
    ans: str = Field(description="Answer of the agent to the query")
    source: List[Source] = Field(description="Source of the agent", default_factory=list)

@tool
def search(query: str) -> str:
    """
    Tool that searches over internet
    :param query: The query to search for
    :return: The search result
    """
    print(f"Searching {query}")
    return tavily.search(query=query)

anthropic_api_key = keyring.get_credential("anthropic", "api_key").password
print(anthropic_api_key)

llm_ollama = ChatOllama(model='gemma3:270m', temperature=0) # doesn't support tool calling
llm_anthropic = ChatAnthropic(
    model='claude-sonnet-4-5',
    temperature=0,
    api_key=anthropic_api_key
)
tools = [TavilySearch()]
agent = create_agent(model=llm_anthropic, tools=tools, response_format=AgentResponse)


def main():
    print("Hello from search-agent!")
    result = agent.invoke({"messages": HumanMessage(content="Search for open job postings for AI Engineer using langchain and langraph in Bengaluru on linkedin")})
    print(result['structured_response'])


if __name__ == "__main__":
    main()
