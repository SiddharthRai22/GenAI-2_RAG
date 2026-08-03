from dotenv import load_dotenv
load_dotenv()
from langchain_mistralai import ChatMistralAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter

# load :- Loads all documents at once.
# step 1
data = PyPDFLoader("document loaders/deeplearning.pdf")
docs = data.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap = 200
)

chunks = splitter.split_documents(docs)

template = ChatPromptTemplate.from_messages([
    ("system", "you are a AI that summarise the text"),
    ("human", "{data}")
])

model = ChatMistralAI(model = "mistral-small-2603")

# prompt = template.format_messages(data = docs[0].page_content)
prompt = template.format_messages(data = docs)

response = model.invoke(prompt)

print(response.content)