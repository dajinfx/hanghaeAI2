import bs4
from langchain import hub
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.document_loaders import PyPDFLoader
import tempfile
import os
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains import ReduceDocumentsChain
from langchain.chains import MapReduceDocumentsChain

class RAGProcessor:
    def __init__(self, model_name,api_key):
        self.model_name = model_name
        self.api_key = api_key
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=300,
            chunk_overlap=50
        )
        self.embeddings = OpenAIEmbeddings(api_key=self.api_key)
        
    def process_pdf(self, uploaded_file):
        try:
            # 创建临时文件保存上传的PDF
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_file_path = tmp_file.name
            
            # 加载PDF
            loader = PyPDFLoader(tmp_file_path)
            docs = loader.load()
            print('논문 페이지 수:', len(docs))
            
        
            # 清理临时文件
            #os.unlink(tmp_file_path)
            print("11")            
            # 获取摘要提示模板
            #prompt = hub.pull("rlm/rag-prompt")  
            print("12")
            #Chroma, Embedding
            splits = self.text_splitter.split_documents(docs)
            print("13")           
            
            print(len(splits))
            print("14")  
            print("1")
            # Map에 사용할 prompt template
            map_template = """다음은 문서 중 일부 내용입니다
            {pages}
            이 문서 목록을 기반으로 주요 내용을 요약해 주세요.

            답변: """

            # Map 프롬프트
            map_prompt = PromptTemplate.from_template(map_template)

            # Map에서 수행할 LLMChain 정의
            llm = ChatOpenAI(model_name=self.model_name, api_key=self.api_key)
            map_chain = LLMChain(llm=llm, prompt=map_prompt)
            print("2")
            # Reduce 단계에서 처리할 프롬프트 정의
            reduce_template = """다음은 요약의 집합입니다:
            {doc_summaries}
            이것들을 바탕으로 통합된 요약을 만들어 주세요.
            요약을 할때,
            1)***
            2)***
            3)***
            4)***
            ....
            이렇게 매개 요약마다 번호를 매겨주세요.  요약수량은 위에서의 4개에 재한하지 않고 객관 요약내용에 따라 조정해주세요.
            답변:"""
            print("3")
            # Reduce 프롬프트 완성
            reduce_prompt = PromptTemplate.from_template(reduce_template)

            # Reduce에서 수행할 LLMChain 정의
            reduce_chain = LLMChain(llm=llm, prompt=reduce_prompt)

            # 문서의 목록을 받아들여, 이를 단일 문자열로 결합하고, 이를 LLMChain에 전달합니다.
            combine_documents_chain = StuffDocumentsChain(
                llm_chain=reduce_chain,
                document_variable_name="doc_summaries" # Reduce 프롬프트에 대입되는 변수
            )
            print("4")
            # Map 문서를 통합하고 순차적으로 Reduce합니다.
            reduce_documents_chain = ReduceDocumentsChain(
                # 호출되는 최종 체인입니다.
                combine_documents_chain=combine_documents_chain,
                # 문서가 `StuffDocumentsChain`의 컨텍스트를 초과하는 경우
                collapse_documents_chain=combine_documents_chain,
                # 문서를 그룹화할 때의 토큰 최대 개수입니다.
                token_max=4000,
            )
            print("5")
            # 문서들에 체인을 매핑하여 결합하고, 그 다음 결과들을 결합합니다.
            map_reduce_chain = MapReduceDocumentsChain(
                # Map 체인
                llm_chain=map_chain,
                # Reduce 체인
                reduce_documents_chain=reduce_documents_chain,
                # 문서를 넣을 llm_chain의 변수 이름(map_template 에 정의된 변수명)
                document_variable_name="pages",
                # 출력에서 매핑 단계의 결과를 반환합니다.
                return_intermediate_steps=False,
            )
            print("Ready for run !")
            # Map-Reduce 체인 실행
            # 입력: 분할된 도큐먼트(②의 결과물)
            result = map_reduce_chain.invoke({"input_documents": splits})
            # 요약결과 출력
            print("Result: ",result)

            return result
            
        except Exception as e:
            print(f"RAG 처리 중 오류 발생: {str(e)}")
            return None
