import os
from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from rich import print

# crating a tool
@tool
def get_text_length(text:str)->int:
    """ return the number of character in a given text"""
    return len(text)

tools = {
    "get_text_length" : get_text_length
}

# llm
llm=ChatOpenAI(
    model="openrouter/free",
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    temperature=0.9,
    max_retries=3,
    timeout=60           
)

# tool binding..
llm_with_tool=llm.bind_tools([get_text_length])
message = []
prompt = input("You: ")
query = HumanMessage(prompt)
message.append(query)

result = llm_with_tool.invoke(message)

message.append(result)

# tools calling..,Execute tool
if result.tool_calls:
    tool_name = result.tool_calls[0]["name"]
    tool_message = tools[tool_name].invoke(result.tool_calls[0])
    message.append(tool_message)
   

result = llm_with_tool.invoke(message)
print(result.content)