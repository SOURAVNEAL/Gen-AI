from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_huggingface import (
    HuggingFaceEmbeddings,
    HuggingFaceEndpoint,
    ChatHuggingFace
)
from langchain_community.vectorstores import FAISS


# --------------------------------------------------
# Load Environment
# --------------------------------------------------

load_dotenv()

# --------------------------------------------------
# Load PDF
# --------------------------------------------------

loader = PyPDFLoader(
    r"C:\Users\admin\Desktop\Langchain_code\RAG_pipeline\LLMbook.pdf"
)

docs = loader.load()

print(f"Pages Loaded : {len(docs)}")

# --------------------------------------------------
# Chunking
# --------------------------------------------------

text_splitter = CharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

all_splits = text_splitter.split_documents(docs)

print(f"Chunks Created : {len(all_splits)}")

# --------------------------------------------------
# Embeddings
# --------------------------------------------------

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# --------------------------------------------------
# Vector Store
# --------------------------------------------------

vectorstore = FAISS.from_documents(
    all_splits,
    embeddings
)

print("Vectorstore created successfully!")

# --------------------------------------------------
# Retriever (MMR Recommended)
# --------------------------------------------------

retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 3,
        "fetch_k": 10
    }
)

# --------------------------------------------------
# LLM
# --------------------------------------------------

llm = HuggingFaceEndpoint(
    repo_id="zai-org/GLM-5.3",
    task="text-generation",
    max_new_tokens=512
)

chat_model = ChatHuggingFace(llm=llm)

# --------------------------------------------------
# User Query
# --------------------------------------------------

query = "What is Large Language Model?"

# --------------------------------------------------
# Retrieve Documents
# --------------------------------------------------

retrieved_docs = retriever.invoke(query)

query_embedding = embeddings.embed_query(query)

print("\nRetrieved Chunks")
print("=" * 80)

context = ""

for idx, doc in enumerate(retrieved_docs, start=1):

    doc_embedding = embeddings.embed_query(
        doc.page_content
    )

    similarity = cosine_similarity(
        [query_embedding],
        [doc_embedding]
    )[0][0]

    print(f"\nChunk {idx}")
    print(f"Cosine Similarity: {similarity*100:.2f}%")
    print("-" * 60)

    print(doc.page_content[:300])

    context += doc.page_content + "\n\n"


# RAG Prompt


prompt = f"""
You are a helpful assistant.

Answer the question ONLY from the provided context.

If the answer is not available in the context,
reply with:

I don't know.

Context:
{context}

Question:
{query}
"""


# Generate Answer

response = chat_model.invoke(prompt)

print("\nGenerated Answer")
print("=" * 80)

print(response.content)