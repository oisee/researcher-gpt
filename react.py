from langchain.agents import load_tools
# from langchain.agents import initialize_agent
# from langchain.agents import AgentType
# from langchain.llms import OpenAI

from langchain.agents import AgentType
from langchain.chat_models import ChatOpenAI
from langchain.agents import initialize_agent

import os
from dotenv import load_dotenv

load_dotenv()

# #llm = OpenAI(temperature=0, model="gpt-3.5-turbo-16k", max_tokens=16000)

# llm = OpenAI(temperature=0)

# agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

# result = agent.run("Who is Leo DiCaprio's girlfriend? What is her current age raised to the 0.43 power?")

# print(result)
from langchain.chat_models import ChatOpenAI
from langchain import OpenAI, Wikipedia
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
from langchain.agents.react.base import DocstoreExplorer
import requests
import json

from langchain.agents.agent_toolkits import PlayWrightBrowserToolkit
from langchain.tools.playwright.utils import (
    create_async_playwright_browser,
    create_sync_playwright_browser, # A synchronous browser is available, though it isn't compatible with jupyter.
)

load_dotenv()
brwoserless_api_key = os.getenv("BROWSERLESS_API_KEY")
serper_api_key = os.getenv("SERP_API_KEY")

docstore = DocstoreExplorer(Wikipedia())

llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-16k", max_tokens=8000)


def search(query):
    url = "https://google.serper.dev/search"

    payload = json.dumps({"q": query})

    headers = {"X-API-KEY": serper_api_key, "Content-Type": "application/json"}

    response = requests.request("POST", url, headers=headers, data=payload)

    print(response.text)

    return response.text


async_browser = create_async_playwright_browser()
browser_toolkit = PlayWrightBrowserToolkit.from_browser(async_browser=async_browser)
tools = browser_toolkit.get_tools()

#tools = load_tools(["serpapi", "llm-math"], llm=llm)

# tools = [
#     Tool(
#         name="Search",
#         func=docstore.search,
#         description="useful for when you need to ask with search",
#     ),
#     # Tool(
#     #     name="Lookup",
#     #     func=docstore.lookup,
#     #     description="useful for when you need to ask with lookup",
#     # )
# ]
# tools = tools + tool0

tools = [Tool(
        name="Intermediate Answer",
        func=search,
        description="Useful for when you need to answer questions about current events, data. You should ask targeted questions.",
    )
]

#llm = OpenAI(temperature=0, model_name="text-davinci-002")
react = initialize_agent(tools, llm, agent=AgentType.OPENAI_MULTI_FUNCTIONS, verbose=True)

question = "Write a research paper on “what is the best option for home Machine Learning)”, providing samples and cost/prices."
react.run(question)

result = react.run(question)
print(result)