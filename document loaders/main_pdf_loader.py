from dotenv import load_dotenv
load_dotenv()
from langchain_mistralai import ChatMistralAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate

# load :- Loads all documents at once.
# step 1
data = PyPDFLoader("document loaders/GRU.pdf")
docs = data.load()

template = ChatPromptTemplate.from_messages([
    ("system", "you are a AI that summarise the text"),
    ("human", "{data}")
])

model = ChatMistralAI(model = "mistral-small-2603")

prompt = template.format_messages(data = docs[0].page_content)
# prompt = template.format_messages(data = docs)

response = model.invoke(prompt)

print(response.content)