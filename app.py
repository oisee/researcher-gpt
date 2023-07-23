import os
from dotenv import load_dotenv
from langchain import Wikipedia
from langchain.agents.react.base import DocstoreExplorer

import tiktoken


from langchain import PromptTemplate
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
from langchain.chat_models import ChatOpenAI
from langchain.chat_models.azure_openai import AzureChatOpenAI
from langchain.prompts import MessagesPlaceholder
from langchain.memory import ConversationSummaryBufferMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Type
from bs4 import BeautifulSoup
import requests
import json
from langchain.schema import SystemMessage
from fastapi import FastAPI
import streamlit as st


from langchain import SerpAPIWrapper
from langchain import LLMMathChain
from langchain.experimental.plan_and_execute import (
    PlanAndExecute,
    load_agent_executor,
    load_chat_planner,
)
from langchain import LLMMathChain


from langchain.agents.agent_toolkits import PlayWrightBrowserToolkit
from langchain.tools.playwright.utils import (
    create_async_playwright_browser,
    create_sync_playwright_browser, # A synchronous browser is available, though it isn't compatible with jupyter.
)

sync_browser = create_sync_playwright_browser()
browser_toolkit = PlayWrightBrowserToolkit.from_browser(sync_browser=sync_browser)
tools = browser_toolkit.get_tools()

docstore = DocstoreExplorer(Wikipedia())

load_dotenv()
brwoserless_api_key = os.getenv("BROWSERLESS_API_KEY")
serper_api_key = os.getenv("SERP_API_KEY")

# 1. Tool for search


def search(query):
    url = "https://google.serper.dev/search"

    payload = json.dumps({"q": query})

    headers = {"X-API-KEY": serper_api_key, "Content-Type": "application/json"}

    response = requests.request("POST", url, headers=headers, data=payload)

    print(response.text)

    return response.text

def clean_text(text):
    return ' '.join(text.split())

enc = tiktoken.encoding_for_model("gpt-3.5-turbo-16k")

# 2. Tool for scraping
def scrape_website(objective: str, url: str):
    # scrape website, and also will summarize the content based on objective if the content is too large
    # objective is the original objective & task that user give to the agent, url is the url of the website to be scraped

    print("Scraping website...")
    # Define the headers for the request
    headers = {
        "Cache-Control": "no-cache",
        "Content-Type": "application/json",
    }

    # Define the data to be sent in the request
    data = {"url": url}

    # Convert Python object to JSON string
    data_json = json.dumps(data)

    # Send the POST request
    post_url = f"https://chrome.browserless.io/content?token={brwoserless_api_key}"
    response = requests.post(post_url, headers=headers, data=data_json)

    # Check the response status code
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, "html.parser")
        text = clean_text( soup.get_text() )
        tokens = len(text)//4
        print(f"--CONTENT, tokens:{tokens}:---\n", text)

        if  tokens > 8000:
            output = summary(objective, text)
            return output
        else:
            return text
    else:
        print(f"HTTP request failed with status code {response.status_code}")


def summary(objective, content):
    llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-16k",max_tokens=8000)
    #llm = OpenAI(temperature=0)
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n"], chunk_size=10000, chunk_overlap=500
    )
    docs = text_splitter.create_documents([content])
    map_prompt = """
    Write a verbose and detailed summary for "[{objective}]", following by short list of key points as bullets, and source urls, max 8000 tokens, based on the following content:
    ### Start of content:

    
    {text}

    
    ### End of content
    SUMMARY:
    """
    combine_prompt = """
    Write a verbose and detailed reasearch paper for "[{objective}]", following by short list of key points as bullets, and source urls, max 12000 tokens, based on the following content:
    ### Start of content:

    
    {text}

    
    ### End of content
    SUMMARY:
    """
    map_prompt_template = PromptTemplate(
        template=map_prompt, input_variables=["text", "objective"]
    )
    combine_prompt_template = PromptTemplate(
        template=combine_prompt, input_variables=["text", "objective"]
    )
    summary_chain = load_summarize_chain(
        llm=llm,
        chain_type="map_reduce",
        map_prompt=map_prompt_template,
        combine_prompt=combine_prompt_template,
        verbose=True,
    )

    output = summary_chain.run(input_documents=docs, objective=objective)

    return output


class ScrapeWebsiteInput(BaseModel):
    """Inputs for scrape_website"""

    objective: str = Field(
        description="The objective & task that users give to the agent."
    )
    url: str = Field(description="The url of the website to be scraped.")


class ScrapeWebsiteTool(BaseTool):
    name = "scrape_website"
    description = "Useful when you need to get data from a website url, passing both url and objective to the function; DO NOT make up any url, the url should only be from the search results."
    args_schema: Type[BaseModel] = ScrapeWebsiteInput

    def _run(self, objective: str, url: str):
        return scrape_website(objective, url)

    def _arun(self, url: str):
        raise NotImplementedError("error here")


# 3. Create langchain agent with the tools above
llm = ChatOpenAI(temperature=0.5, model="gpt-3.5-turbo-16k",max_tokens=8000)
llm_math_chain = LLMMathChain.from_llm(llm=llm, verbose=True)

tools = tools + [
    Tool(
        name="Search_Google",
        func=search,
        description="Useful for when you need to answer questions about current events, data. You should ask targeted questions.",
    ),
    ScrapeWebsiteTool(),
    Tool(
        name="Calculator",
        func=llm_math_chain.run,
        description="useful for when you need to answer questions about math",
    ),
    Tool(
        name="Search_Wikipedia",
        func=docstore.search,
        description="useful for when you need to ask facts from Wikipedia",
    ),
    Tool(
        name="Lookup_Wikipedia",
        func=docstore.lookup,
        description="useful for when you need to ask with lookup on documents found in Wikipedia",
    )
]

# def fake_func(inp: str) -> str:
#     return "Amazing number: " + inp + " very cool and unique!"

# fake_tools = [
#     Tool(
#         name=f"About number {i}",
#         func=fake_func,
#         description=f"Useful function about number {i}",
#     )
#     for i in range(99)
# ]

# tools = tools + fake_tools

for tool in tools:
    print(tool.name, tool.description)

system_message = SystemMessage(
    content="""You are a world class researcher, who can do detailed research on any topic and produce facts based results.
            You do not make things up, you will try as hard as possible to gather facts & data to back up the research.
            You will use the tools below to do research on the objective.
            You will Explain all the steps you take to do the research, and the result of the research.
            You will Reflect on the result, and think about what you can do to improve the research quality.
            You will do 5 to 7 iterations of research.

            For all the found options you will do the detailed explanation which will be sub-research, explaining application or meaning of the option to the objective, and the result of the sub-research.
            For all the found options you will do the detailed explanation which will be sub-research, explaining application or meaning of the option to the objective, and the result of the sub-research.


            Please make sure you complete the objective above with the following rules:
            1. You should do enough research to gather as much information as possible about the objective.
            "Search_Google" - Useful for when you need to answer questions about current events, data. You should ask targeted questions.
            2. If there are URL of relevant links & articles, you will use navigate_browser function Navigate a browser to the specified URL.
            "Search_Wikipedia" for the objective, and use the "Lookup Wikipedia" function to lookup the objective in the Wikipedia database.
            "extract_text" to Extract all the text on the current webpage
            "extract_hyperlinks" Extract all hyperlinks on the current webpage
            3. After browser & search, you should think "is there any new things I should search & scraping based on the data I collected to increase research quality?" If answer is yes, continue; Do from 5 to 7 iterations.
            4. Reflect on result.
            5. You should not make things up, you should only write facts & data that you have gathered. 
            6. Try to visit at least first 5 links of the search results, and browser the content of the website to gather more information.
            7. You should not write anything that is not related to the objective.
            8. In the final output, You should include all reference data & links to back up your research; You should include all reference data & links to back up your research.
            9. In the final output, You should include all reference data & links to back up your research; You should include all reference data & links to back up your research.
            """
)

agent_kwargs = {
    "extra_prompt_messages": [MessagesPlaceholder(variable_name="memory")],
    "system_message": system_message,
}

llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-16k",max_tokens=8000)
memory = ConversationSummaryBufferMemory(
    memory_key="memory", return_messages=True, llm=llm, max_token_limit=8000)

agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.OPENAI_FUNCTIONS,
    verbose=True,
    agent_kwargs=agent_kwargs,
    memory=memory,
)


# 4. Use streamlit to create a web app
def main():
    st.set_page_config(page_title="AI research agent", page_icon=":bird:")

    st.header("AI research agent :bird:")
    query = st.text_input("Research goal")

    if query:
        st.write("Doing research for ", query)

        result = agent({"input": query})

        st.info(result['output'])


if __name__ == '__main__':
    main()
