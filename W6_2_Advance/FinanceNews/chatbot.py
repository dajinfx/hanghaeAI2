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
        #print("API 키:", self.api_key)
        try:
            if model_config['model_type'] == 'openai':
                self.model = ChatOpenAI(model=model_config['model_name'], api_key=self.api_key)
            elif model_config['model_type'] == 'anthropic':
                self.model = "" #ChatAnthropic(...)
            elif model_config['model_type'] == 'deepseek':
                self.model = DeepSeekChat(
                    api_key=self.api_key,
                    model=model_config['model_name']
            )
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
        # 메시지 목록을 역순으로 정렬하여 새 메시지가 하단에 표시되도록 함
        messages = reversed(st.session_state.messages)
        for message in messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                
    def process_user_input(self, prompt):
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

    def generate_response(self):
        with st.chat_message("assistant"):
            messages = []
            
            # 시스템 메시지
            system_message = SystemMessage(content="You are a helpful AI assistant. You can analyze documents and answer questions about their content.")
            messages.append(system_message)
            
            # 문서 내용이 있으면 컨텍스트에 추가합니다.
            if 'document_content' in st.session_state:
                context_message = SystemMessage(content=f"Document content: {st.session_state['document_content'][:2000]}")
                messages.append(context_message)
            
            # 대화 기록
            for m in st.session_state.messages:
                content = str(m.get("content", ""))
                if m["role"] == "user":
                    messages.append(HumanMessage(content=content))
                elif m["role"] == "assistant":
                    messages.append(AIMessage(content=content))

            try:
                print("전송할 메시지:", messages)
                result = self.model.invoke(messages)
                print("원본 결과:", result)
                response = str(result.content)  
                st.markdown(response)

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response
                })
            except Exception as e:
                error_msg = f"응답 생성 오류: {str(e)}"
                print(error_msg)
                st.error(error_msg)
                import traceback
                print("전체 오류 추적:", traceback.format_exc())
            
    def hide_streamlit_elements(self):
        # Streamlit 기본 요소 숨기기
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
        
    def run(self):
        self.initialize_session()
        
        # 채팅 기록과 입력창을 배치할 컬럼 생성
        chat_column = st.container()
        
        with chat_column:
            # 입력창을 최하단에 배치
            user_input = st.chat_input("What is up?")
            
            # 채팅 기록 표시 (이전 메시지는 위에, 새 메시지는 아래에)
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
            
            # 사용자 입력 처리
            if user_input:
                self.process_user_input(user_input)
                self.generate_response()
                
            self.hide_streamlit_elements()
            #print("API 키:", self.api_key)

class DeepSeekChat:
    def __init__(self, api_key, model):
        self.api_key = api_key
        self.model = model
        self.base_url = "https://api.deepseek.com/v1"

    def invoke(self, messages):
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        formatted_messages = []
        for msg in messages:
            if isinstance(msg, SystemMessage):
                formatted_messages.append({"role": "system", "content": msg.content})
            elif isinstance(msg, HumanMessage):
                formatted_messages.append({"role": "user", "content": msg.content})
            elif isinstance(msg, AIMessage):
                formatted_messages.append({"role": "assistant", "content": msg.content})

        data = {
            "model": self.model,
            "messages": formatted_messages,
            "temperature": 0.7,
            "max_tokens": 2000
        }

        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=data
            )
            
            if response.status_code == 200:
                result = response.json()
                return type('Response', (), {'content': result['choices'][0]['message']['content']})
            else:
                error_msg = response.json().get('error', {}).get('message', 'Unknown error')
                raise Exception(f"API call failed: {error_msg}")
                
        except:
            raise Exception(f"Network error: {error_msg}")