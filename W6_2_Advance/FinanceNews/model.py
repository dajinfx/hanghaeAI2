import streamlit as st
import os
from dotenv import load_dotenv
import platform

class MyModel:
    def __init__(self, model_name):
        self.model_name = model_name
        self.model_configs = {
            'gpt-4o-mini': {
                'env_key': 'gpt-4o-mini',
                'model_type': 'openai',
                'model_name': 'gpt-4o-mini'
            },
            'deepseek-chat': {
                'env_key': 'deepseek-chat',
                'model_type': 'deepseek',
                'model_name': 'deepseek-chat'
            },
            'claude': {
                'env_key': 'claude_api',
                'model_type': 'anthropic',
                'model_name': 'claude-3-opus-20240229'
            }
        }
        
    def get_api(self):
        if self.model_name not in self.model_configs:
            raise ValueError(f"Unsupported model: {self.model_name}")
        config = self.model_configs[self.model_name]
        
        print("mymodel-config: ",config)   
        #api_key = os.environ.get(config['env_key'])
        #print("mymodel-config['env_key']: ",config['env_key'])  

        #print("mymodel-api_key: ",api_key)   
        #print(f"Getting API key for {config['env_key']}: {api_key}")

        env_path = os.path.join(os.getcwd(), ".env")
        load_dotenv(env_path)  # 加载 .env 文件
        api_key = os.getenv(config['env_key'])
        print(f"--------  API_KEY: {api_key}")



        return api_key
    
    def get_model_config(self):
        if self.model_name not in self.model_configs:
            raise ValueError(f"Unsupported model: {self.model_name}")
            
        return self.model_configs[self.model_name]
    






    