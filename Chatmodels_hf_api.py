from dotenv import load_dotenv
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

load_dotenv()

## CREATE The HuggingFaceEndpoint object with the model repo_id and task
llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen3.8-2.4T-A95B",
    task="text-generation"
)
#pass the llm to the ChatHuggingFace class to create a chat model
chat_model = ChatHuggingFace(llm=llm)

#call the invoke method of the chat model to get the response from the model
response = chat_model.invoke("What is the capital of West Bengal?")
print(response.content)