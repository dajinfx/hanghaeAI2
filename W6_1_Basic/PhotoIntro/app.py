# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 17:49:22 2024

@author: jjin
"""

from chatbot import ChatBot
from typing_extensions import Annotated
from streamlit_pandas_profiling import st_profile_report
from ydata_profiling import ProfileReport
import streamlit as st
from streamlit_option_menu import option_menu
from PIL import Image

def setup_sidebar():
    with st.sidebar:
        st.title("Image Upload")
        
        # 이미지 버튼튼
        uploaded_file = st.file_uploader("Choose an image", type=['png', 'jpg', 'jpeg'])
        
        if uploaded_file is not None and len(st.session_state.uploaded_images) < 5:
            try:
                image = Image.open(uploaded_file)
                if st.button("Add Image"):
                    st.session_state.uploaded_images.append(image)
                    st.session_state.image_names.append(uploaded_file.name)
                    st.success(f"Added image: {uploaded_file.name}")
            except Exception as e:
                st.error(f"Error loading image: {str(e)}")
        
        # 이미지 Side bar에다 리스트 
        if st.session_state.image_names:
            st.write("Uploaded Images:")
            for i, name in enumerate(st.session_state.image_names):
                col1, col2 = st.columns([4, 1])
                with col1:
                    st.write(f"{i+1}. {name}")
                with col2:
                    if st.button("❌", key=f"delete_{i}"):
                        st.session_state.uploaded_images.pop(i)
                        st.session_state.image_names.pop(i)
                        st.rerun()
        
        # 모든 이미지 삭제하기기
        if st.session_state.uploaded_images:
            if st.button("Clear All Images"):
                st.session_state.uploaded_images = []
                st.session_state.image_names = []
                st.rerun()
    

def main(app_title='GPT Bot', model_name='gpt-4o-mini'):
    # 초기화 session state
    if 'uploaded_images' not in st.session_state:
        st.session_state.uploaded_images = []
        st.session_state.image_names = []
    
    # Side bar
    setup_sidebar()
    
    # 챗보가져오기 chatbot
    chatbot = ChatBot(app_title, model_name)
    chatbot.run()

if __name__ == "__main__":
    app_title = "My Bot"
    model_name = "gpt-4o-mini"
    main(app_title, model_name)


