# PyPDFLoader use for pdf file
from langchain_community.document_loaders import PyPDFLoader
# RecursiveCharacterTextSplitter use for Splitting recursively
from langchain_text_splitters import RecursiveCharacterTextSplitter

data = PyPDFLoader("splitter/GRU.pdf")

docs = data.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap = 10
)

chunks = splitter.split_documents(docs)

# print(chunks)
print(chunks[0].page_content)
# print(len(chunks))