import os
from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel,RunnableLambda


# RunnableParallel
simple_prompt=ChatPromptTemplate.from_template(
    "Explain {topic} in 1-2 lines"
)
detailed_prompt=ChatPromptTemplate.from_template(
    "Explain {topic} in detail"
)

model=ChatOpenAI(
    model="openrouter/free",
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    temperature=0.9,
    max_retries=3,
    timeout=60           
)

parser = StrOutputParser()

topic = "Machine Learning"

chain=RunnableParallel({
    "short":simple_prompt|model|parser,
    "detailed":detailed_prompt|model|parser
})
result=chain.invoke({"topic":"Machine Learning"})

print(result['short'])
print(result['detailed'])


# RunnableLambda
simple_prompt=ChatPromptTemplate.from_template(
    "Explain {topic} in 1-2 lines"
)
detailes_prompt=ChatPromptTemplate.from_template(
    "Explain {topic} in detail"
)

model=ChatOpenAI(
    model="openrouter/free",
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    temperature=0.9,
    max_retries=3,
    timeout=60           
)

parser = StrOutputParser()

topic = "Machine Learning"

chain=RunnableParallel({
    "short":RunnableLambda(lambda x :x['short']) |simple_prompt|model|parser ,
    "detailes":RunnableLambda(lambda x :x['detailes']) |detailes_prompt|model|parser
})

result=chain.invoke({
    "short":{"topic":"Machine Learning"},
    "detailes":{"topic":"Deep Learning"}
})

print(result['short'])
print(result['detailes'])