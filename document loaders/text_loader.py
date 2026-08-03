# TextLoader use for text file
from langchain_community.document_loaders import TextLoader

# file path
data = TextLoader("document loaders/notes.txt")

# load :- Loads all documents at once.
docs = data.load()

print(docs) # data and meatdata
print(docs[0]) # first(zero) index data and meatdata
print(docs[0].page_content) # first(zero) index data
print(len(docs)) # print length of docs