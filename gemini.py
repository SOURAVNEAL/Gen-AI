import os
import numpy as np
import streamlit as st

from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter

from langchain_huggingface import HuggingFaceEmbeddings

from langchain_google_genai import ChatGoogleGenerativeAI

from langchain_community.vectorstores import FAISS

# --------------------------------------------------
# Load Environment
# --------------------------------------------------
# Requires GOOGLE_API_KEY in your .env file, e.g.:
#   GOOGLE_API_KEY=your-gemini-api-key
# Get a free key at https://aistudio.google.com/apikey

load_dotenv()

# --------------------------------------------------
# Page Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="PDF QA Chatbot"
)

st.title("RAG-based PDF Question Answering")

st.write(
    "Ask questions from your PDF document."
)

# --------------------------------------------------
# Load RAG Pipeline
# --------------------------------------------------

@st.cache_resource
def load_rag_pipeline():

    # --------------------------------------------------
    # Load PDF
    # --------------------------------------------------

    loader = PyPDFLoader(
        r"C:\Users\admin\Desktop\Langchain_code\RAG_pipeline\Computer Networks - A Tanenbaum - 5th edition.pdf"
    )

    docs = loader.load()

    # --------------------------------------------------
    # Text Splitter
    # --------------------------------------------------

    text_splitter = CharacterTextSplitter(
        chunk_size=100,
        chunk_overlap=20
    )

    all_splits = text_splitter.split_documents(docs)

    # --------------------------------------------------
    # HuggingFace Embeddings
    # --------------------------------------------------

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # --------------------------------------------------
    # FAISS Vector Store
    # --------------------------------------------------

    vectorstore = FAISS.from_documents(
        all_splits,
        embeddings
    )

    # --------------------------------------------------
    # Retriever
    # --------------------------------------------------

    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 3,
            "fetch_k": 10
        }
    )

    # --------------------------------------------------
    # Gemini LLM
    # --------------------------------------------------
    # NOTE: Reads GOOGLE_API_KEY from the environment (.env).
    # Get a free-tier key at https://aistudio.google.com/apikey
    #
    # Using a Flash-Lite model here on purpose: the full "Flash"
    # models (gemini-3.8-flash, 3.7, 3.6, 3.5, and the
    # "gemini-flash-latest" alias, which currently resolves to one
    # of these) are limited to only ~20 free requests/day, which is
    # why you hit RESOURCE_EXHAUSTED / 429 after a handful of
    # questions. The Flash-Lite tier gets a much larger free daily
    # quota (currently ~500 requests/day), so it's a better fit for
    # iterative testing on the free tier. Quality is slightly lower
    # than full Flash, but plenty for straightforward RAG QA.
    #
    # If you need full-Flash quality and are willing to pay, switch
    # back to "gemini-flash-latest" or a pinned Flash model and
    # enable billing on your Google Cloud project for higher limits.
    # Current free-tier numbers: https://ai.google.dev/gemini-api/docs/rate-limits

    chat_model = ChatGoogleGenerativeAI(
        model="gemini-3.1-flash-lite",
        temperature=0.1,
        max_output_tokens=512
    )

    return retriever, embeddings, chat_model


retriever, embeddings, chat_model = load_rag_pipeline()

# --------------------------------------------------
# User Question
# --------------------------------------------------

query = st.text_input(
    "Enter your question"
)

# --------------------------------------------------
# Answer Button
# --------------------------------------------------

if st.button("Get Answer"):

    if not query.strip():

        st.warning(
            "Please enter a question."
        )

        st.stop()

    # ==================================================
    # RETRIEVAL
    # ==================================================

    retrieved_docs = retriever.invoke(
        query
    )

    # --------------------------------------------------
    # Query Embedding
    # --------------------------------------------------

    query_embedding = embeddings.embed_query(
        query
    )

    context = ""

    # --------------------------------------------------
    # Display Retrieved Chunks
    # --------------------------------------------------

    st.subheader(
        "Retrieved Chunks"
    )

    chunk_similarities = []
    chunk_embeddings = []

    for idx, doc in enumerate(
        retrieved_docs,
        start=1
    ):

        doc_embedding = embeddings.embed_query(
            doc.page_content
        )

        similarity = cosine_similarity(
            [query_embedding],
            [doc_embedding]
        )[0][0]

        chunk_similarities.append(similarity)
        chunk_embeddings.append(doc_embedding)

        st.write(
            f"Chunk {idx} | "
            f"Similarity: {similarity * 100:.2f}%"
        )

        st.text(
            doc.page_content[:300]
        )

        context += (
            doc.page_content +
            "\n\n"
        )

    # ==================================================
    # RAG PROMPT
    # ==================================================

    prompt = f"""
You are a helpful assistant.

Answer the question ONLY from the provided context.

If the answer is not available in the context,
reply exactly:

I don't know.

Context:

{context}

Question:

{query}

Answer:
"""

    # ==================================================
    # GENERATE ANSWER
    # ==================================================

    response = chat_model.invoke(
        prompt
    )

    # --------------------------------------------------
    # Extract plain text from the response
    # --------------------------------------------------
    # NOTE: Some Gemini models (e.g. gemini-3.x with extended
    # thinking) return `response.content` as a LIST of content
    # blocks (e.g. [{"type": "text", "text": "...", "extras": {...}}])
    # instead of a plain string, to carry thought signatures.
    # `response.text` is LangChain's own accessor that already
    # normalizes this -- it returns the plain string for both
    # Gemini 2.5-and-earlier (plain string content) and Gemini 3.x
    # (list-of-blocks content), joining only the actual text blocks
    # and skipping reasoning/signature data. Manually parsing
    # response.content ourselves was fragile against the exact
    # block shape, which is why the answer stopped showing --
    # response.text is the version-proof way to get it.

    answer = getattr(
        response,
        "text",
        None
    )

    # response.text can be a plain string or, in some versions, a
    # bound method -- call it if so. Fall back to str(content) if
    # .text isn't usable for some reason, so we never crash here.

    if callable(answer):
        answer = answer()

    if not answer:
        answer = str(response.content)

    # --------------------------------------------------
    # Display Answer
    # --------------------------------------------------

    st.subheader(
        "Answer"
    )

    st.write(
        answer
    )

    # ==================================================
    # RAG EVALUATION (non-LLM, embedding-based metrics)
    # ==================================================
    # NOTE: The previous version used Ragas metrics
    # (faithfulness, answer_relevancy, context_precision) which
    # internally make extra LLM calls to the HF endpoint to judge
    # the answer. That's slow, costs extra API calls/quota, and
    # is prone to "nan" results when the judge LLM's output can't
    # be parsed.
    #
    # These three metrics below are computed purely from cosine
    # similarity between embeddings (the same local
    # sentence-transformers model already loaded for retrieval).
    # No additional LLM/API calls are made.

    st.subheader(
        "RAG Evaluation"
    )

    try:

        # --------------------------------------------------
        # Answer Embedding
        # --------------------------------------------------

        answer_embedding = embeddings.embed_query(
            answer
        )

        # --------------------------------------------------
        # Metric 1: Answer Relevancy (proxy)
        # --------------------------------------------------
        # How semantically close the answer is to the question.
        # Ragas' LLM-based answer_relevancy generates synthetic
        # questions from the answer and compares them to the real
        # question -- this direct embedding comparison is a
        # lightweight, LLM-free stand-in for the same idea.

        answer_relevancy_score = cosine_similarity(
            [answer_embedding],
            [query_embedding]
        )[0][0]

        # --------------------------------------------------
        # Metric 2: Faithfulness (proxy)
        # --------------------------------------------------
        # How semantically grounded the answer is in the
        # retrieved context, measured as the max cosine similarity
        # between the answer and any individual retrieved chunk.
        # Ragas' LLM-based faithfulness breaks the answer into
        # statements and checks each against the context with an
        # LLM judge; this embedding-similarity version approximates
        # "is the answer close to something actually in the
        # context" without needing a judge LLM.

        faithfulness_scores = cosine_similarity(
            [answer_embedding],
            chunk_embeddings
        )[0]

        faithfulness_score = float(
            np.max(faithfulness_scores)
        )

        # --------------------------------------------------
        # Metric 3: Context Precision (proxy)
        # --------------------------------------------------
        # Average query-to-chunk similarity across the retrieved
        # chunks (already computed above while displaying each
        # chunk) -- a simple, LLM-free measure of how relevant the
        # retrieved chunks are to the question.

        context_precision_score = float(
            np.mean(chunk_similarities)
        )

        # --------------------------------------------------
        # Display Scores
        # --------------------------------------------------

        st.caption(
            "Computed locally from embeddings only -- "
            "no extra LLM/API calls."
        )

        col1, col2, col3 = st.columns(3)

        with col1:

            st.metric(
                "Answer Relevancy",
                f"{answer_relevancy_score:.3f}"
            )

        with col2:

            st.metric(
                "Faithfulness",
                f"{faithfulness_score:.3f}"
            )

        with col3:

            st.metric(
                "Context Precision",
                f"{context_precision_score:.3f}"
            )

        with st.expander("Debug: raw similarity scores"):

            st.write(
                {
                    "answer_relevancy (answer vs question)": answer_relevancy_score,
                    "faithfulness (max answer vs chunk)": faithfulness_score,
                    "per_chunk_faithfulness": [
                        float(s) for s in faithfulness_scores
                    ],
                    "context_precision (mean question vs chunk)": context_precision_score,
                    "per_chunk_context_similarity": [
                        float(s) for s in chunk_similarities
                    ],
                }
            )

    except Exception as e:

        st.error(
            f"RAG evaluation failed: {e}"
        )