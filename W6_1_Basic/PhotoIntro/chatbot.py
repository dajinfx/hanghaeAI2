import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from model import MyModel
import os
import requests
import io
import base64
import traceback

class ChatBot:
    def __init__(self, app_title, model_name='gpt-4o-mini'):
        self.model_handler = MyModel(model_name)
        self.api_key = self.model_handler.get_api()
        self.title = app_title
        
        model_config = self.model_handler.get_model_config()
        print("Model Config:", model_config)
        #print("API Key:", self.api_key)
        
        os.environ['OPENAI_API_KEY'] = self.api_key
        
        if model_config['model_type'] == 'openai':
            self.model = ChatOpenAI(model=model_config['model_name'], api_key=self.api_key)
        elif model_config['model_type'] == 'anthropic':
            self.model = "" #ChatAnthropic(...)
        elif model_config['model_type'] == 'deepseek':
            self.model = DeepSeekChat(
                api_key=self.api_key,
                model=model_config['model_name']
            )

    def display_images(self):
        if st.session_state.uploaded_images:
            cols = st.columns(5)
            for i, image in enumerate(st.session_state.uploaded_images):
                with cols[i]:
                    st.image(image, use_container_width=True)

    def encode_image(self, image):
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')

    def compare_images(self):
        if len(st.session_state.uploaded_images) < 2:
            st.warning("Please upload at least 2 images to compare")
            return
        
        image_contents = [self.encode_image(img) for img in st.session_state.uploaded_images]
        
        prompt = f"""I have {len(st.session_state.uploaded_images)} images to compare. Please analyze these images and tell me:
        1. 이 이미지들 사이의 주요 차이점은 무엇입니까?
        2. 유사점은 무엇입니까?
        3. 어느 이미지가 품질이나 구성이 더 좋습니까?
        자세한 분석을 제공해 주십시오."""
        
        with st.chat_message("user"):
            st.markdown(prompt)
            for i, img in enumerate(st.session_state.uploaded_images):
                st.image(img, caption=f"Image {i+1}", width=200)
        
        st.session_state.messages.append({
            "role": "user",
            "content": prompt,
            "images": image_contents
        })
        
        self.generate_response()

    def initialize_session(self):
        if "messages" not in st.session_state:
            st.session_state.messages = []
            
    def display_chat_history(self):
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
    
    def generate_response(self):
        with st.chat_message("assistant"):
            try:
                if len(st.session_state.uploaded_images) > 0:
                    # 이미지 준비
                    image_messages = []
                    for img in st.session_state.uploaded_images:
                        base64_image = self.encode_image(img)
                        image_messages.append({
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        })
                    
                    # 프롬프트 
                    messages = [
                        {
                            "role": "system",
                            "content": "당신은 이미지 분석을 하는 AI 전문가입니다. 자세하고 정확한 답변을 제공해 주세요."
                        },
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": st.session_state.messages[-1]["content"]},
                                *image_messages
                            ]
                        }
                    ]
                    
                    # GPT-4 Vision API
                    response = self.model.invoke(messages)
                    st.markdown(response.content)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response.content
                    })
                else:
                    st.warning("Please upload some images first.")
                
            except Exception as e:
                st.error(f"Error generating response: {str(e)}")
                print("Full traceback:", traceback.format_exc())

    # 우측상단 streamlit 아이콘 없애기        
    def hide_streamlit_elements(self):
        st.markdown(
            r"""
            <style>
            #MainMenu {visibility: hidden;}
            .stDeployButton {visibility: hidden;}
            .stAppToolbar{visibility: hidden;}
            </style>
            """, unsafe_allow_html=True    
        )
        
    def run(self):
        st.title(self.title)
        self.initialize_session()
        
        # 메인화면에 이미지 보이기
        self.display_images()
        
        # 비고 필요성을 검정
        if hasattr(st.session_state, 'compare_clicked') and st.session_state.compare_clicked:
            self.compare_images()
            st.session_state.compare_clicked = False
        
        # 대화 history display
        self.display_chat_history()
        
        # 쳇봇 대화입력
        if prompt := st.chat_input("Ask about the images or type any question"):
            self.process_user_input(prompt)
            self.generate_response()
            
        self.hide_streamlit_elements()

    def process_user_input(self, prompt):
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # 메시지 history 저장
        st.session_state.messages.append({
            "role": "user",
            "content": prompt
        })

#DeepSeek 를 테스트하려다 Access 이슈로 개발 일시 중단상태입니다.
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
                
        except requests.exceptions.RequestException as e:
            raise Exception(f"Network error: {str(e)}")