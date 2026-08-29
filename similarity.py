from huggingface_hub import InferenceClient
from sklearn.metrics.pairwise import cosine_similarity

client = InferenceClient()

documents = [
     "Microsoft develops innovative software, cloud computing, AI.",
    "Google provides internet-based products",
    "IBM delivers enterprise technology, artificial intelligence, cloud computing, consulting services."
]

query = "Tell me about Google"

query_emb = client.feature_extraction(
    query,
    model="sentence-transformers/all-MiniLM-L6-v2"
)

best_doc = ""
best_score = -1

for doc in documents:
    doc_emb = client.feature_extraction(
        doc,
        model="sentence-transformers/all-MiniLM-L6-v2"
    )

    score = cosine_similarity([query_emb], [doc_emb])[0][0]

    if score > best_score:
        best_score = score
        best_doc = doc

print("Query :", query)
print("Answer:", best_doc)