import os
import re
import time
from datetime import datetime, timedelta
import pandas as pd
from typing import List, Dict
from sentence_transformers import CrossEncoder
from langchain_google_genai import GoogleGenerativeAIEmbeddings 
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
load_dotenv()

# --- Cấu hình ---
faiss_folder = "vectors/faiss_index"  # FAISS database từ main.py
cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

# --- Single API Key (no rotation/limits here) ---
API_KEY = os.getenv("GOOGLE_API_KEY_1") or os.getenv("GOOGLE_API_KEY")

# --- Khởi tạo Embedding model dùng 1 key ---
embedding_model = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=API_KEY
)


choice_map = {
    0: "Public transports",
    1: "Private modes",
    2: "Soft modes",
}


# --- Load FAISS index ---
db = FAISS.load_local(
    faiss_folder,
    embeddings=embedding_model,
    allow_dangerous_deserialization=True
)


# --- FAISS-based Retrieval Functions ---
def get_similar_vectors_by_id(query_id: int, query: str, examples_id: int = 2) -> List[Dict]:
    """Lấy tất cả vectors có cùng ID với query_id, re-rank theo query, trả về top n."""
    # Thu thập tất cả Document có cùng ID
    collected_docs = []
    for value in db.docstore._dict.values():
        items = value if isinstance(value, (list, tuple)) else [value]
        for item in items:
            doc = item[0] if isinstance(item, tuple) else item
            if hasattr(doc, "metadata") and doc.metadata.get("id") == query_id:
                collected_docs.append(doc)

    if not collected_docs:
        return []
    if len(collected_docs) > examples_id:
        # 🔹 Cross-Encoder re-ranking theo query
        pairs = [(query, d.page_content) for d in collected_docs]
        scores = cross_encoder.predict(pairs)
        ranked = sorted(zip(collected_docs, scores), key=lambda x: x[1], reverse=True)
        top_docs = [doc for doc, _ in ranked[:examples_id]]
    else:
        top_docs = collected_docs

    final_situations = []
    for i, doc in enumerate(top_docs, 1):
        content = getattr(doc, "page_content", "") or ""
        choice = doc.metadata.get("choice", 0)
        choice_desc = choice_map.get(choice, "unknown")
        # Tìm và chỉ lấy phần thông tin về chuyến đi, loại bỏ thông tin cá nhân
        match = re.search(r"(.*?free of charge)", content, re.DOTALL)
        result = match.group(1) if match else "No trip information found"
        example = (
            f"Situation {i}: {result}. That person chose {choice_desc}. "
        )
        final_situations.append(example)
    return final_situations

def balanced_retrieval_with_rerank(query: str, query_id: int, k_per_label: int = 5, top_k: int = 5):
    """
    Balanced Retrieval theo label + Cross-Encoder re-ranking.
    Trả về: 1 chuỗi text gồm các ví dụ.
    """
    query_vector = embedding_model.embed_query(query)
    candidates = []
    
    # 🔹 Balanced Retrieval
    hits = db.similarity_search_by_vector(query_vector, k=100)

    # Lọc bỏ các document có id trùng với query_id
    hits = [doc for doc in hits if doc.metadata.get("id") != query_id]

    labels = set(doc.metadata.get("choice") for doc in hits)
    for label in labels:
        docs_label = [doc for doc in hits if doc.metadata.get("choice") == label]
        candidates.extend(docs_label[:k_per_label])

    # 🔹 Cross-Encoder re-ranking
    pairs = [(query, cand.page_content) for cand in candidates]
    scores = cross_encoder.predict(pairs)
    ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
    final_docs = [doc for doc, s in ranked[:top_k]]

    examples = []
    for i, doc in enumerate(final_docs, 1):
        choice = doc.metadata.get("choice", 0)
        choice_desc = choice_map.get(choice, "unknown")
        example = (
            f"Example {i}: {doc.page_content} Their choice was {choice_desc}. "
        )
        examples.append(example)

    return examples


def retrieval(query: str, id: int):
    situations = get_similar_vectors_by_id(id, query)
    examples = balanced_retrieval_with_rerank(query, id)
    return situations, examples

#test
if __name__ == "__main__":
    pd = pd.read_csv("data/test.csv")
    query = pd.iloc[1]["INFOR"]
    id = pd.iloc[1]["ID"]
    situations, examples = retrieval(query, id)
    print(query)
    print(id)
    
    print("Situations:")
    for doc in situations:
       print(doc)
    print("Examples:")
    for doc in examples:
        print(doc)






