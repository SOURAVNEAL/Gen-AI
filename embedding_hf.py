from dotenv import load_dotenv
from huggingface_hub import InferenceClient

load_dotenv()

#Create the InferenceClient object to interact with the Hugging Face Inference API
client = InferenceClient()

#This method sends the text to a Hugging Face embedding model and asks:"Convert this text into a list of numbers that represents its meaning."
#model="sentence-transformers/all-MiniLM-L6-v2" is used to get the embeddings for the text
document = "Microsoft, founded in 1975 by Bill Gates and Paul Allen, started as a software company with MS-DOS and later Windows."
embeddings = client.feature_extraction(
    document,
    model="sentence-transformers/all-MiniLM-L6-v2"
)

print(len(embeddings))
print(embeddings[:10])