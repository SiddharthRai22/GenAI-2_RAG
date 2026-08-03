#load pdf 
#split into chunks 
#create the embeddings 
#store into chroma 
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings 
from langchain_community.vectorstores import Chroma 
from dotenv import load_dotenv

load_dotenv()

# load :- Loads all documents at once.
# step 1
data = PyPDFLoader("document loaders/deeplearning.pdf")
docs = data.load()

# step 2
splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap = 200
)

chunks = splitter.split_documents(docs)

# step 3
embedding_model = OpenAIEmbeddings()

# step 4
vectorstore = Chroma.from_documents(
    documents= chunks,
    embedding=embedding_model,
    persist_directory="chroma_db"
)