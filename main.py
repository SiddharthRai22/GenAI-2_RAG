# from dotenv import load_dotenv
# load_dotenv()
# from langchain_mistralai import ChatMistralAI
# from langchain_community.document_loaders import PyPDFLoader
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_text_splitters import RecursiveCharacterTextSplitter

# template = ChatPromptTemplate.from_messages([
#     ("system", "you are a AI that summarise the text"),
#     ("human", "{data}")
# ])

# model = ChatMistralAI(model = "mistral-small-2603")










from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

# user query embedding
embedding_model = OpenAIEmbeddings()

vectorstore = Chroma(
    persist_directory= "chroma_db",
    embedding_function=embedding_model
)

retriever = vectorstore.as_retriever(
    search_type = "mmr",
    search_kwargs = {
        "k" : 4,
        "fetch_k":10, # first you do similarity search and take out 10 result and then do mmr on that 10 result and take out 4 result
        "lambda_mult" :0.5 # lambda_mult is use for diverse result 1 mean less diverse and 0 mean more diverse
    }
)

llm = ChatMistralAI(model = "mistral-small-2506")

#prompt template 
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a helpful AI assistant.

Use ONLY the provided context to answer the question.

If the answer is not present in the context,
say: "I could not find the answer in the document."
"""
        ),
        (
            "human",
            """Context:
{context}

Question:
{question}
"""
        )
    ]
)

print("Rag system created ")

print("press 0 to exit ")

while True:
    query = input("You : ")
    if query == "0":
        break 
    
    docs = retriever.invoke(query)

    context = "\n\n".join(
        [doc.page_content for doc in docs]
    )
    
    final_prompt = prompt.invoke({
        "context" :context,
        "question": query
    })
    
    response = llm.invoke(final_prompt)

    print(f"\n AI: {response.content}")
    