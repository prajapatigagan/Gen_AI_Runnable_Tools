import os
import requests
from dotenv import load_dotenv
load_dotenv()

from langchain_mistralai import ChatMistralAI
from langchain.tools import tool
from langchain_core.messages import HumanMessage, ToolMessage
from tavily import TavilyClient
from rich import print
from langchain.agents import create_agent 
from langchain.agents.middleware import wrap_tool_call

# Now lets a create some tools

# weather tool

@tool
def get_weather(city :str)->str:
    """Get Current weather of a city"""
    api_key = os.getenv("OPENWEATHER_API_KEY")
    url=f"http://api.openweathermap.org/data/2.5/weather?q={city},IN&appid={api_key}&units=metric"

    response=requests.get(url)
    data=response.json()

    print("DEBUG:",data)

    if str(data.get("cod")) != "200":
        return f"Error: {data.get('message', 'Could not fetch weather')}"

    temp=data["main"]["temp"]
    desc=data["weather"][0]["description"]

    return f"Weather in {city}:{desc},{temp}°C"


# Tavity new tool

tavily_client=TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

@tool
def get_news(city:str)->str:
    """ get latest news about the city"""

    response = tavily_client.search(
        query=f"latest news in {city}",
        search_depth="basic",
        max_results=3
    )
    
    results = response.get("results", [])
    
    if not results:
        return f"No news found for {city}"
    
    news_list = []
    
    for r in results:
        title = r.get("title", "No title")
        url = r.get("url", "")
        snippet = r.get("content", "")
        
        news_list.append(
            f"- {title}\n  🔗 {url}\n  📝 {snippet[:100]}..."
        )
    
    return f"Latest news in {city}:\n\n" + "\n\n".join(news_list)

print(get_news.invoke("Jhansi"))

# llm

llm=ChatOpenAI(
    model="openrouter/free",
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    temperature=0.9,
    max_retries=3,
    timeout=60           
)

tool={
    "get_weather":get_weather,
    "get_news":get_news
}
llm_with_tool=llm.bin.bind_tools([get_weather,get_news])

#Agent LOOP -Very imp
messages=[]

print("City Agent | type exit to quit")