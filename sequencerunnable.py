import os
from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

prompt=ChatPromptTemplate.from_template(
    "Explain {topic} in simple words"
)
model=ChatOpenAI(
    model="openrouter/free",
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    temperature=0.9,
    max_retries=3,
    timeout=60           
)

parser=StrOutputParser()

# runnable used
runnable=prompt|model|parser
result=runnable.invoke("Machine Learning")

print(result)