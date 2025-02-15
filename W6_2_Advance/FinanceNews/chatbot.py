# -*- coding: utf-8 -*-
"""
생성일: 2025년 1월 30일 17:49:22

작성자: jjin
"""

import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from model import MyModel
import os
from dotenv import load_dotenv
import requests

class ChatBot:
    def __init__(self, app_title, model_name='gpt-4o-mini'):
        self.model_handler = MyModel(model_name)
        self.api_key = self.model_handler.get_api()
        self.title = app_title
        
        model_config = self.model_handler.get_model_config()
        print("모델 설정:", model_config)
        print("api_key", self.api_key)
        
        try:
            if model_config['model_type'] == 'openai':
                self.model = ChatOpenAI(model=model_config['model_name'], api_key=self.api_key)
        except:
            st.session_state.messages = []
            st.session_state.messages.append({
                "role": "assistant",
                "content": "API key error"
            })

    def initialize_session(self):
        if "messages" not in st.session_state:
            st.session_state.messages = []
            
    def display_chat_history(self):
        messages = reversed(st.session_state.messages)
        for message in messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                
    def process_user_input(self, prompt):
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

    async def generate_response(self):
        with st.chat_message("assistant"):
            messages = []
            
            # System message
            system_message = SystemMessage(content="You are a helpful AI assistant.")
            messages.append(system_message)
            
            # Add user messages to the context
            for m in st.session_state.messages:
                content = str(m.get("content", ""))
                if m["role"] == "user":
                    messages.append(HumanMessage(content=content))
                elif m["role"] == "assistant":
                    messages.append(AIMessage(content=content))

            try:
                print("Sending messages:", messages)  # Debugging output
                result = self.model.invoke(messages)  # Ensure this is awaited
                print("Raw result:", result)  # Debugging output
                response = str(result.content)  
                st.markdown(response)

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response
                })
            except Exception as e:
                error_msg = f"Error generating response: {str(e)}"
                print(error_msg)  # Debugging output
                st.error(error_msg)
            
    def hide_streamlit_elements(self):
        st.markdown(
            r"""
            <style>
            #MainMenu {visibility: hidden;}
            .stDeployButton {
                    visibility: hidden;
                }
            .stAppToolbar{
                    visibility: hidden;
                }
            </style>
            """, unsafe_allow_html=True    
        )
        
    async def run_chatbot(self):
        """Run the chatbot logic."""
        self.initialize_session()
        
        chat_column = st.container()
        
        with chat_column:
            user_input = st.chat_input("What is up?")
            
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
            
            if user_input:
                self.process_user_input(user_input)
                await self.generate_response()  # Await the response generation
