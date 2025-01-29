from langchain.schema import Document
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter

docs1 = [
    Document(
        metadata={"source": "example.txt"},
        page_content="이는 긴 문서이므로 처리를 위해 여러 섹션으로 나눌 수 있습니다." * 10
    )
]

# 문장길이
print(f"문장길이: {len(docs1[0].page_content)}"+"\n")

# CharacterTextSplitter 강제분리
text_splitter1 = CharacterTextSplitter(chunk_size=30, chunk_overlap=3)
splits1 = text_splitter1.split_documents(docs1)

print("CharacterTextSplitter -----------------------------------------------------")

print(f"CharacterTextSplitter 분리수: {len(splits1)}")
for i, split in enumerate(splits1):
    print(f"Chunk {i+1}: {split.page_content}")

print("\n"+"RecursiveCharacterTextSplitter -----------------------------------------------------")
# RecursiveCharacterTextSplitter
text_splitter2 = RecursiveCharacterTextSplitter(chunk_size=30, chunk_overlap=3)
splits2 = text_splitter2.split_documents(docs1)

print(f"RecursiveCharacterTextSplitter 분리수: {len(splits2)}")
for i, split in enumerate(splits2):
    print(f"Chunk {i+1}: {split.page_content}")