from huggingface_hub import InferenceClient
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_text_splitters import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEndpoint
from langchain_community.vectorstores import FAISS
from langchain.tools import tool
from langchain.agents import create_agent
from huggingface_hub import InferenceClient
from dotenv import load_dotenv

client = InferenceClient()
load_dotenv()

from langchain_huggingface import HuggingFaceEmbeddings

loader=PyPDFLoader (r"C:\Users\admin\Desktop\Langchain_code\RAG_pipeline\LLMbook.pdf")

docs =loader.load()
#print(len(docs ))


## CREATE The HuggingFaceEndpoint object with the model repo_id and task
llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen3.8-2.4T-A95B",
    task="text-generation"
)
#pass the llm to the ChatHuggingFace class to create a chat model
chat_model = ChatHuggingFace(llm=llm)

### CHUNKING THE DOCUMENTS
textsplitter=CharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200)
all_splits=textsplitter.split_documents(docs)

# print(len(all_splits))
#printll_splits[0].page_content)

### Embedding the chunks and creating a vectorstore
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectorstore = FAISS.from_documents(all_splits, embeddings)

print("Vectorstore created successfully!")

@tool
def retrieve(query: str):
    docss= vectorstore.similarity_search(query,k=3)
    data=[]
    return docss

prompt = "You are a helpful assistant that answers questions based on the context provided. If the answer is not in the context, say 'I don't know'."
agent = create_agent(
    model=chat_model,
    tools=[retrieve],
    system_message=prompt
)
query = "What is the main topic of the document?"
result = agent.run(query)
print(result)
