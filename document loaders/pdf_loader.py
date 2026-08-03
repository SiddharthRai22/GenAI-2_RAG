# PyPDFLoader use for pdf file
from langchain_community.document_loaders import PyPDFLoader

data = PyPDFLoader("document loaders/GRU.pdf")

docs = data.load()

print(docs)
print(docs[14])
print(len(docs))