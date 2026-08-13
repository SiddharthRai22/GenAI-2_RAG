# TextLoader use for text file
from langchain_community.document_loaders import TextLoader
# CharacterTextSplitter use for Splitting by character
from langchain_text_splitters import CharacterTextSplitter

# default :- separator = "\n\n"

splitter = CharacterTextSplitter(
    separator= "",
    chunk_size = 10,
    chunk_overlap = 1
)

# file path
data = TextLoader("splitter/notes.txt")

# load :- Loads all documents at once.
docs = data.load()

chunks = splitter.split_documents(docs)

print(chunks)
# print(chunks[0].page_content)
# print(len(chunks)) # print length of chunks

#(extra) see chunk text not necessary for code
# for i in chunks:
#     print(i.page_content)
#     print()