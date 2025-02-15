import bs4
from langchain import hub
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import PyPDFLoader
import tempfile
import os
import asyncio
from typing import Optional

class RAGProcessor:
    def __init__(self, model_name, api_key):
        self.model_name = model_name
        self.api_key = api_key
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=300,
            chunk_overlap=50
        )

   
    async def web_url(self, content) -> Optional[str]:
        try:
            # 문서 분할
            chunk_size = 100  # Define the size of each chunk
            splits = []
            
            # Split the content into chunks
            for i in range(0, len(content), chunk_size):
                splits.append(content[i:i + chunk_size])

            print(f"분할된 청크 수: {len(splits)}")

            # LLM 설정
            llm = ChatOpenAI(model_name=self.model_name, api_key=self.api_key)
            print("-- url 1")

            # 요약 프롬프트 생성
            map_prompt = PromptTemplate.from_template(
                """다음은 문서 중 일부 내용입니다:
                {content}
                이 문서의 주요 내용을 요약해 주세요.
                답변: """
            )
            print("-- url 2")

            # 통합 프롬프트 생성
            reduce_prompt = PromptTemplate.from_template(
                """다음은 여러 요약들입니다:
                {summaries}
                이것들을 바탕으로 통합된 요약을 만들어 주세요.
                요약을 할때,
                1)***
                2)***
                3)***
                4)***
                ....
                이렇게 매개 요약마다 번호를 매겨주세요. 
                요약수량은 위에서의 4개에 재한하지 않고 객관 요약내용에 따라 조정해주세요.
                답변: """
            )
            print("-- url 3")

            # 처리 그래프 생성
            summarize_doc = (
                RunnableParallel(content=lambda x: x)  # Change this line to handle string directly
                | map_prompt
                | llm
                | StrOutputParser()
            )
            print("-- url 4")

            # 모든 문서 조각 처리
            summaries = []
            for doc in splits:
                summary = await summarize_doc.ainvoke(doc)
                summaries.append(summary)
            print("-- url 5")

            # 모든 요약 통합
            final_chain = reduce_prompt | llm | StrOutputParser()
            result = await final_chain.ainvoke({"summaries": "\n".join(summaries)})
            print("-- url  6")

            print("result:", result)
            return result

        except Exception as e:
            print(f"RAG 처리 중 오류 발생: {str(e)}")
            return None        
      

        
    async def process_pdf(self, uploaded_file) -> Optional[str]:
        try:
            # PDF 파일을 임시 저장
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_file_path = tmp_file.name
            
            # PDF 로드
            loader = PyPDFLoader(tmp_file_path)
            docs = loader.load()
            print('논문 페이지 수:', len(docs))
            
            # 문서 분할
            splits = self.text_splitter.split_documents(docs)
            print(f"분할된 청크 수: {len(splits)}")

            # LLM 설정
            llm = ChatOpenAI(model_name=self.model_name, api_key=self.api_key)
            print("-- 1")

            # 요약 프롬프트 생성
            map_prompt = PromptTemplate.from_template(
                """다음은 문서 중 일부 내용입니다:
                {content}
                이 문서의 주요 내용을 요약해 주세요.
                답변: """
            )
            print("-- 2")

            # 통합 프롬프트 생성
            reduce_prompt = PromptTemplate.from_template(
                """다음은 여러 요약들입니다:
                {summaries}
                이것들을 바탕으로 통합된 요약을 만들어 주세요.
                요약을 할때,
                1)***
                2)***
                3)***
                4)***
                ....
                이렇게 매개 요약마다 번호를 매겨주세요. 
                요약수량은 위에서의 4개에 재한하지 않고 객관 요약내용에 따라 조정해주세요.
                답변: """
            )
            print("-- 3")

            # 처리 그래프 생성
            summarize_doc = (
                RunnableParallel(content=lambda x: x.page_content)
                | map_prompt
                | llm
                | StrOutputParser()
            )
            print("-- 4")

            # 모든 문서 조각 처리
            summaries = []
            for doc in splits:
                summary = await summarize_doc.ainvoke(doc)
                summaries.append(summary)
            print("-- 5")

            # 모든 요약 통합
            final_chain = reduce_prompt | llm | StrOutputParser()
            result = await final_chain.ainvoke({"summaries": "\n".join(summaries)})
            print("-- 6")

            # 임시 파일 정리
            os.unlink(tmp_file_path)
            
            print("result:", result)
            return result

        except Exception as e:
            print(f"RAG 처리 중 오류 발생: {str(e)}")
            return None
