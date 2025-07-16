from dotenv import load_dotenv
load_dotenv()

import os
key = os.getenv("OPENAI_API_KEY")
api_base_url = os.getenv("OPENAI_API_BASE")

from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage


print(key, api_base_url)

llm = ChatOpenAI(
    model="openai/gpt-3.5-turbo-16k",
    temperature=0.7,
    openai_api_key=key,
    openai_api_base=api_base_url,
    
)

response = llm([HumanMessage(content="Hello! What can you do?")])
print(response)

