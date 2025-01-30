# -*- coding: utf-8 -*-
"""
생성일: 2024년 4월 4일 17:49:22

작성자: jjin
"""

from chatbot import ChatBot
import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
from io import StringIO
import docx
import PyPDF2
import io
from langchain.document_loaders import PyPDFLoader
import tempfile
import os
from model import MyModel
from rag import RAGProcessor

import bs4
from langchain import hub
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain


def sidebar():
    """사이드바를 구성하고 사용자 설정을 반환합니다."""
    with st.sidebar.expander("🔎 Model", expanded=True):
        st.image("https://www.onepointltd.com/wp-content/uploads/2024/02/shutterstock_1166533285-Converted-02.png")
        st.title("Document Analysis")

    with st.sidebar.expander("🔎 Model", expanded=True):
        model_cat = st.selectbox(
            'Model', ('OpenAI-4o', 'Deepseek', 'Grok'), index=0)

    with st.sidebar.expander("🔎 TOPIC", expanded=True):
        topic = st.text_input(
            'Keywords or phrases to search in the News', 'Bitcoin')
        topic = topic.strip()
        if not topic:
            st.warning('Please enter a valid topic.')

    with st.sidebar.expander("🔝 TOP-STORIES", expanded=True):
        category = st.selectbox(
            'Category', ('Crypto', 'Forex', 'Stock', 'Fund'), index=0)

    return model_cat, topic, category

def read_pdf(file):
    pdf_reader = PyPDF2.PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() + "\n"
    return text

def read_docx(file):
    doc = docx.Document(file)
    text = ""
    for para in doc.paragraphs:
        text += para.text + "\n"
    return text

def read_txt(file):
    return str(file.read(), "utf-8")

def main(app_title='GPT Bot', model_name='gpt-4o-mini'):
    st.set_page_config(layout="wide")
    model_handler = MyModel(model_name)
    api_key = model_handler.get_api()
    
    # 创建 RAG 处理器
    rag_processor = RAGProcessor(model_name,api_key)
    
    # 左侧边栏
    with st.sidebar:
        model_cat, topic, category = sidebar()
    
    # 创建两列布局：主内容区和右侧 AI Assistant
    main_col, right_col = st.columns([3, 1])
    
    # 主内容区
    with main_col:
        st.title("Document Analysis")
        
        # 文件上传部分
        uploaded_file = st.file_uploader("Choose a file", type=['txt', 'pdf', 'docx'])
        
        if uploaded_file is not None:
            # 显示文件信息
            file_details = {
                "Filename": uploaded_file.name,
                "FileType": uploaded_file.type,
                "FileSize": f"{uploaded_file.size / 1024:.2f} KB"
            }
            st.write("### File Details")
            for key, value in file_details.items():
                st.write(f"- {key}: {value}")
            
            # 读取文件内容
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
                    st.write("### File Content Preview")
                    #st.text_area("원본 내용", content[:1000] + "...", height=300)
                    
                    #st.session_state['document_content'] = content
                    
                    # 处理 PDF 文件
                    if uploaded_file.type == "application/pdf":
                        with st.spinner('문서 분석 중...'):
                            result = rag_processor.process_pdf(uploaded_file)
                            
                            if result:
                                st.write("### 문서 분석 결과")
                                st.write("### 문서 요약")
                                st.write(result)
            except Exception as e:
                st.error(f"Error reading file: {str(e)}")
    
    # 右侧 AI Assistant
    with right_col:
        st.title("AI Assistant")
        chatbot = ChatBot(app_title, model_name)
        chatbot.run()

if __name__ == "__main__":
    app_title = "My Bot"
    model_name = "gpt-4o-mini"  
    main(app_title, model_name)




