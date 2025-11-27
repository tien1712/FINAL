import os
import pandas as pd
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
load_dotenv()

# --- Config ---
CSV_PATH = "data/train.csv"       # file của bạn
FAISS_FOLDER = "vectors/faiss_index"

# --- API Key (chỉ dùng 1 key) ---
API_KEY = os.getenv("GOOGLE_API_KEY_1")

# --- Load data ---
df = pd.read_csv(CSV_PATH)
texts = df['INFOR'].fillna("").astype(str).tolist()
metadatas = [{"id": int(row["ID"]), "choice": int(row["CHOICE"])} for _, row in df.iterrows()]

# --- Khởi tạo model embeddings dùng 1 key ---
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=API_KEY
)

# --- Build FAISS in batches (không giới hạn rate) ---
def build_faiss_in_batches(texts, metadatas, embeddings_model, faiss_folder, batch_size=20):
    db = None
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        batch_metas = metadatas[i:i+batch_size]
        try:
            batch_db = FAISS.from_texts(
                texts=batch_texts,
                embedding=embeddings_model,
                metadatas=batch_metas
            )
            if db is None:
                db = batch_db
            else:
                db.merge_from(batch_db)
            print(f"✅ Processed {i+len(batch_texts)}/{len(texts)} rows")
        except Exception as e:
            print(f"❌ Lỗi tại batch {i//batch_size + 1}: {str(e)}")
            continue
    if db is not None:
        db.save_local(faiss_folder)
        print(f"✅ FAISS index saved to '{faiss_folder}'")
    else:
        print("❌ Không thể tạo FAISS index")
    return db

# --- Run ---
if __name__ == "__main__":
    print(f"🚀 Bắt đầu xử lý {len(texts)} texts với 1 API key")
    db = build_faiss_in_batches(texts, metadatas, embeddings, FAISS_FOLDER, batch_size=10)
