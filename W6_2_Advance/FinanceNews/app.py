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
import asyncio
from streamlit.runtime.scriptrunner import add_script_run_ctx


def sidebar():
    """사이드바를 구성하고 사용자 설정을 반환합니다."""
    with st.sidebar.expander("🔎 Model", expanded=True):
        st.image("https://www.onepointltd.com/wp-content/uploads/2024/02/shutterstock_1166533285-Converted-02.png")
        st.title("Document Analysis")

    with st.sidebar.expander("🔎 Model", expanded=True):
        model_cat = st.selectbox(
            'Model', ('OpenAI-4o', 'Deepseek', 'Grok'), index=0)

    #with st.sidebar.expander("🔎 TOPIC", expanded=True):
    #    topic = st.text_input(
    #        'Keywords or phrases to search in the News', 'Bitcoin')
    #    topic = topic.strip()
    #    if not topic:
    #        st.warning('Please enter a valid topic.')

    #with st.sidebar.expander("🔝 TOP-STORIES", expanded=True):
    #    category = st.selectbox(
    #        'Category', ('Crypto', 'Forex', 'Stock', 'Fund'), index=0)

    return model_cat, None, None

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
    
    # RAG 프로세서 생성
    rag_processor = RAGProcessor(model_name, api_key)
    
    # 좌측 사이드바
    with st.sidebar:
        model_cat, topic, category = sidebar()
    
    # 두 열 레이아웃 생성
    main_col, right_col = st.columns([3, 1])
    
    # 우측 AI Assistant (먼저 렌더링하여 독립성 보장)
    assistant_container = right_col.container()
    with assistant_container:
        st.title("AI Assistant")
        chatbot = ChatBot(app_title, model_name)
        chatbot.run()
    
    # 메인 콘텐츠 영역
    main_container = main_col.container()
    with main_container:
        st.title("Document Analysis")
        
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
                            
                            result = asyncio.run(process_pdf_async())
                            if result:
                                st.write("### 문서 요약")
                                st.write(result)
                    
            except Exception as e:
                st.error(f"Error reading file: {str(e)}")

if __name__ == "__main__":
    app_title = "My Bot"
    model_name = "gpt-4o-mini"  
    main(app_title, model_name)




