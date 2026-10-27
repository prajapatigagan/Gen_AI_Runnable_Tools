import os
from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel,RunnablePassthrough

model=ChatOpenAI(
    model="openrouter/free",
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    temperature=0.9,
    max_retries=3,
    timeout=60           
)

parser = StrOutputParser()

code_prompt=ChatPromptTemplate.from_messages([
    ("system", "You are a helpful coding assistant."),
    ("human", "{topic}")
]
)

explain_prompt=ChatPromptTemplate.from_messages([
    ("system","You are a helpful assistant who explains code in simple terms"),
    ("human","Explain the following code in simple words:\n{code}")
])

seq1=code_prompt|model|parser

seq2 = RunnableParallel(
    {"code" :  RunnablePassthrough(),
     "explanation" : explain_prompt | model | parser
    }
)

chain=seq1|seq2

result=chain.invoke({"topic":"write a code of palindrom in python"})

print(result['code'])
print(result['explanation'])