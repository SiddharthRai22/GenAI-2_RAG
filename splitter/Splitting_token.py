# PyPDFLoader use for pdf file
from langchain_community.document_loaders import PyPDFLoader
# TokenTextSplitter use for Splitting by token
from langchain_text_splitters import TokenTextSplitter

data = PyPDFLoader("splitter/GRU.pdf")

docs = data.load()

splitter = TokenTextSplitter(
    chunk_size = 1000,
    chunk_overlap = 10
)

chunks = splitter.split_documents(docs)

# print(chunks)
print(chunks[0].page_content)
# print(len(chunks))