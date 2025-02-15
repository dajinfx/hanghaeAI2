# -*- coding: utf-8 -*-
"""
생성일: 2025년 2월 1일 17:49:22

작성자: jjin
"""

from chatbot import ChatBot
import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
from io import StringIO
import PyPDF2
import io
from langchain.document_loaders import PyPDFLoader
import tempfile
import os
from model import MyModel
from rag import RAGProcessor

import bs4
from langchain import hub
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import asyncio
from streamlit.runtime.scriptrunner import add_script_run_ctx
import platform
from dotenv import load_dotenv, set_key
import os
import docx2txt
import requests
from bs4 import BeautifulSoup

# Define content as a global variable
content = ""

def sidebar():
    """사이드바를 구성하고 사용자 설정을 반환합니다."""
    
    # Add custom CSS to change sidebar color, input text color, border color, and expander border color
    # st.markdown(
    # """
    #     <style>
    #     /* Change sidebar background color to black */
    #     .stSidebar {
    #         background-color: black !important;  
    #         color: white;  
    #         border: 2px solid blue !important;  /* Set sidebar border color to blue */
    #     }
    #     
    #     /* Change text color in the sidebar to white */
    #     .stSidebar * {
    #         color: white !important;  
    #     }
    #     
    #     /* Change text color in the input fields to gray */
    #     .stTextInput input {
    #         color: gray !important;  /* Set default font color to gray */
    #         background-color: black !important;  /* Change background to black */
    #     }
    #     
    #     /* Change border color of each expander in the sidebar to blue */
    #     .stExpander {
    #         border: 2px solid blue !important;  /* Set expander border color to blue */
    #     }
    #     </style>
    # """,
    # unsafe_allow_html=True
    # )
    
    with st.sidebar.expander("🔎 Model", expanded=True):
        model_cat = st.selectbox('', ('gpt-4o-mini', 'Deepseek', 'Grok'), index=0)
          
    
    # API Key 설정
    with st.sidebar.expander("🔑 API Key", expanded=False):
        #key_file = st.file_uploader("Upload key file", type=['txt'])
        #key_content = st.session_state['api_key']
        key_file = st.file_uploader("Upload key file", type=['txt'], accept_multiple_files=False)
        if key_file is not None:
            key_content = key_file.getvalue().decode('utf-8').strip()
            env_para = model_cat

            #os.environ[env_para] = key_content
            #print ("key_content: ",key_content)
            #st.success("Key loaded!")

            env_path = os.path.join(os.getcwd(), ".env")
            save_env_variable(env_path, model_cat, key_content)
            #print(f"{env_path} env. 입력성공！")
            
        if 'api_key' in st.session_state:
            st.write("Key status: ✅")
            if st.button("Clear key"):
                del st.session_state['api_key']
                st.rerun()

    print ("model_cat: ",model_cat)
    
    # Replace the topic input with checkboxes
    with st.sidebar.expander("🔎 TOPIC", expanded=True):
        topics = ['Forex', 'Crypto', 'Stock', 'Fund']
        selected_topics = []
        for topic in topics:
            if st.checkbox(topic):
                selected_topics.append(topic)
        if not selected_topics:
            st.warning('Please select at least one topic.')
        
    return model_cat, selected_topics, None

# 手动写入 .env 文件
def save_env_variable(env_path,key, value):
    env_lines = []
    key_exists = False

    if os.path.exists(env_path):
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip().startswith(f"{key}="):
                    env_lines.append(f"{key}={value}\n")
                    key_exists = True
                else:
                    env_lines.append(line)

    if not key_exists:
        env_lines.append(f"{key}={value}\n")

    with open(env_path, "w", encoding="utf-8") as f:
        f.writelines(env_lines)

    print(f"{key}={value} 已成功写入 {env_path}")

def read_pdf(file):
    pdf_reader = PyPDF2.PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() + "\n"
    return text

def read_docx(file):
    return docx2txt.process(file)

def read_txt(file):
    return file.getvalue().decode('utf-8')

def fetch_content(url):
    """주어진 URL에서 내용을 가져옵니다."""
    try:
        response = requests.get(url)
        response.raise_for_status()  # 요청이 성공했는지 확인
        soup = BeautifulSoup(response.text, 'html.parser')
        # 본문 내용을 추출 (예: <p> 태그의 텍스트)
        paragraphs = soup.find_all('p')
        content = ' '.join([para.get_text() for para in paragraphs])
        return content
    except Exception as e:
        st.error(f"Error fetching content: {e}")
        return ""
    
async def main(app_title='GPT Bot', model_name='gpt-4o-mini'):
    st.set_page_config(layout="wide")
    
    model_handler = MyModel(model_name)
    
    # session_state에서 api_key 가져오기
    api_key = st.session_state.get('api_key', model_handler.get_api())
    
    # RAG 프로세서 생성
    rag_processor = RAGProcessor(model_name, api_key)
    
    # 좌측 사이드바
    with st.sidebar:
        model_cat, topics, category = sidebar()
    
    # 두 열 레이아웃 생성
    main_col, right_col = st.columns([3, 1])
    
    # 우측 AI Assistant (먼저 렌더링하여 독립성 보장)
    assistant_container = right_col.container()
    with assistant_container:
        chatbot = ChatBot(app_title, model_name)
        await chatbot.run_chatbot()  # Run the chatbot asynchronously

    # 메인 콘텐츠 영역
    main_container = main_col.container()
    with main_container:
        global content  # Declare content as a global variable
        if content == "":
            content = ""  # Initialize content if it's empty

        url = st.text_input("URL을 입력하세요:")
        if st.button("내용 가져오기"):
            content = fetch_content(url)
            print("Gather url content: ", "-" * 20)
            print(content)
            
            # Display the content
            st.write("### Fetched Content")

        # 파일 업로드 섹션
        uploaded_file = st.file_uploader("Choose a file", type=['txt', 'pdf', 'docx'])
        
        if uploaded_file is not None:
            # 파일 정보 표시
            file_details = {
                "Filename": uploaded_file.name,
                "FileType": uploaded_file.type,
                "FileSize": f"{uploaded_file.size / 1024:.2f} KB"
            }
            st.write("### File Details")
            for key, value in file_details.items():
                st.write(f"- {key}: {value}")
            
            # 파일 내용 읽기
            try:
                if uploaded_file.type == "text/plain":
                    content = read_txt(uploaded_file)
                elif uploaded_file.type == "application/pdf":
                    content = read_pdf(uploaded_file)
                elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                    content = read_docx(uploaded_file)
                else:
                    st.error("Unsupported file type")
                    content = None
                
                if content:
                    st.write("### File Content Summary")
                    
                    # PDF 파일 처리
                    if uploaded_file.type == "application/pdf":
                        with st.spinner('문서 분석 중...'):
                            async def process_pdf_async():
                                return await rag_processor.process_pdf(uploaded_file)
                            
                            result = await process_pdf_async()
                            if result:
                                st.write("### 문서 요약")
                                st.write(result)
                    
            except:
                st.error(f"Error reading file")

        if content:
            #st.write(content)
            async def process_pdf_async():
                return await rag_processor.web_url(content)
            result = await process_pdf_async()
            if result:
                st.write("### 문서 요약")
                st.write(result)
                            






if __name__ == "__main__":
    app_title = "My Bot"
    model_name = "gpt-4o-mini"  
    asyncio.run(main(app_title, model_name))




