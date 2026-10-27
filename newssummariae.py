import os
from dotenv import load_dotenv
load_dotenv()

from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

search_tool=TavilySearchResults(max_result=5)

llm=model=ChatOpenAI(
    model="openrouter/free",
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    temperature=0.9,
    max_retries=3,
    timeout=60           
)

parser=StrOutputParser()

prompt=ChatPromptTemplate.from_template(
    """
you are a helpful assistant
summarize the following news inti clear bullet points
{news}
"""
)

chain=prompt|model|parser

news_result=search_tool.run("Latest AI news of 2026")

result=chain.invoke({"news": news_result})

print(result)

print(search_tool.description)
print(search_tool.name)
print(search_tool.args)