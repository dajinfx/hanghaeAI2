# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 22:42:36 2025

@author: jingu
"""
from langchain_deepseek import ChatDeepSeekAI
import os


api_key_2 = os.environ["openAI_gpt-4o-mini"]

print("api_key 2: ",api_key_2)

api_key_1 = os.environ["deepseek_api_key"]

print("api_key 1: ",api_key_1)



llm = ChatDeepSeekAI(
    model="deepseek-chat",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    api_key=api_key_1
    # other params...
)


response = llm.predict("Translate 'I love programming' to French.")
print(response)



