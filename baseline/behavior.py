import os, json, re, ast
import asyncio
import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm
from langchain_google_genai import ChatGoogleGenerativeAI
from prompt_behavior import prompt
import csv
# code xử lý tiếp các bản ghi có prediction rỗng
load_dotenv()

# Lấy duy nhất 1 API key từ biến môi trường GOOGLE_API_KEY_1
api_key = os.getenv("GOOGLE_API_KEY_1")
if not api_key or not str(api_key).strip():
    raise RuntimeError("Không tìm thấy API key trong biến môi trường GOOGLE_API_KEY_1")

# Đọc dữ liệu gốc
df = pd.read_csv("data/behavior.csv")  # lưu lại chỉ số dòng gốc

# Đọc/khởi tạo kết quả hiện tại
result_path = "behavior/behavior.csv"
process_all = False
try:
    result_df = pd.read_csv(result_path)
    if 'ID' not in result_df.columns or 'response' not in result_df.columns:
        raise ValueError("behavior.csv không đúng định dạng")
    # Tìm các bản ghi có response rỗng
    empty_responses = result_df[result_df['response'].isna() | (result_df['response'] == '')]
    print(f"Tìm thấy {len(empty_responses)} bản ghi có response rỗng")
    if len(empty_responses) == 0:
        # Nếu không còn rỗng, không cần xử lý thêm
        print("Không có bản ghi nào cần xử lý lại!")
        # Không exit để có thể hỗ trợ chạy toàn tập nếu người dùng xóa file rồi chạy lại
        rows_to_process = []
    else:
        empty_ids = empty_responses['ID'].tolist()
        rows_to_process = df[df['ID'].isin(empty_ids)].to_dict(orient="records")
        print(f"Sẽ xử lý lại {len(rows_to_process)} bản ghi")
except Exception:
    # File chưa tồn tại hoặc không hợp lệ → khởi tạo mới và xử lý toàn bộ test
    os.makedirs(os.path.dirname(result_path), exist_ok=True)
    base = {
        'ID': df['ID'],
        'response': [''] * len(df)
    }
    result_df = pd.DataFrame(base)
    result_df.to_csv(result_path, index=False)
    rows_to_process = df.to_dict(orient="records")
    process_all = True
    print(f"Khởi tạo {result_path}. Sẽ xử lý toàn bộ {len(rows_to_process)} bản ghi trong behavior.csv")

def extract_behavior_response(raw: str):
    """Extract phần response sau cụm từ 'transportation behavior:'"""
    if raw is None or not str(raw).strip():
        return raw
    
    text = str(raw).strip()
    # Tìm cụm từ "transportation behavior:" (case-insensitive)
    pattern = r"transportation behavior\s*:"
    match = re.search(pattern, text, re.IGNORECASE)
    
    if match:
        # Lấy phần sau cụm từ này
        start_pos = match.end()
        extracted = text[start_pos:].strip()
        # Loại bỏ khoảng trắng và dòng trống đầu tiên
        extracted = extracted.lstrip('\n').strip()
        return extracted if extracted else raw
    
    # Nếu không tìm thấy, trả về toàn bộ response
    return raw

def safe_parse_json(raw: str):
    if raw is None or not str(raw).strip():
        raise ValueError("Empty model response")
    text = str(raw).strip()
    try:
        return json.loads(text)
    except Exception:
        # Cố gắng tìm đoạn JSON trong chuỗi dài
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            candidate = text[start:end + 1]
            # Thử parse JSON chuẩn trước
            try:
                return json.loads(candidate)
            except Exception:
                pass
            # Fallback: parse kiểu dict Python với nháy đơn bằng ast.literal_eval
            try:
                data = ast.literal_eval(candidate)
                if isinstance(data, dict):
                    return data
            except Exception:
                pass
        # Fallback cuối: thử literal_eval toàn bộ văn bản
        try:
            data = ast.literal_eval(text)
            if isinstance(data, dict):
                return data
        except Exception:
            pass

async def call_model_async(index, total, row, api_key, retries=2, delay=30):
    # Tạo model theo key mỗi lần gọi để tránh xung đột state giữa threads
    def _do_invoke(prompt_text: str):
        model = ChatGoogleGenerativeAI(model="gemini-2.5-pro", google_api_key=api_key)
        return model.invoke(prompt_text)

    for attempt in range(retries):
        try:
            prompt_text = prompt(row)
            print(f"[{index+1}/{total}] 🔍 Đang xử lý lại row id={row['ID']}...")
            response = await asyncio.to_thread(_do_invoke, prompt_text)
            raw = str(getattr(response, "content", response))

            # Lưu toàn bộ response vào file theo ID
            response_dir = "results/responses"
            os.makedirs(response_dir, exist_ok=True)
            response_file = os.path.join(response_dir, f"{row['ID']}.txt")
            with open(response_file, "w", encoding="utf-8") as f:
                f.write(raw)
            print(f"[{index+1}] 💾 Đã lưu response vào {response_file}")

            # Extract phần response sau "transportation behavior:"
            extracted_response = extract_behavior_response(raw)

            print(f"[{index+1}] ✅ Hoàn tất row id={row['ID']}")
            return row["ID"], extracted_response
        except Exception as e:
            if ("429" in str(e) or "rate" in str(e).lower()) and attempt < retries - 1:
                print(f"[{index+1}] ⏳ Lỗi 429/rate limit. Đợi {delay}s rồi thử lại...")
                await asyncio.sleep(delay)
            else:
                print(f"[{index+1}] ❌ Lỗi ở row id={row.get('ID', 'N/A')}: {e}")
                with open("errors.log", "a") as f:
                    f.write(f"Lỗi ở dòng {index} (id={row.get('ID', 'N/A')}): {e}\n")
                return row.get("ID", index), None

async def process_with_single_key(index, total, row, api_key, retries=5, delay=30):
    """Xử lý 1 request với 1 key, retry tối đa 5 lần nếu lỗi"""
    print(f"[{index+1}] 🔑 Dùng API key, thử tối đa {retries} lần...")
    result = await call_model_async(index, total, row, api_key, retries=retries, delay=delay)
    return result

async def worker(name, api_key, jobs_q: asyncio.Queue, result_df, result_path: str, total: int, df_lock: asyncio.Lock, progress_state: dict, progress_lock: asyncio.Lock):
    while True:
        item = await jobs_q.get()
        if item is None:
            jobs_q.task_done()
            break

        try:
            # Xử lý request với 1 key, retry 5 lần nếu lỗi
            result = await process_with_single_key(item[0], total, item[1], api_key, retries=5, delay=30)
            # result = (id, raw_response)
            # cập nhật kết quả và lưu file an toàn
            async with df_lock:
                if result[1] is not None:
                    result_df.loc[result_df['ID'] == result[0], 'response'] = result[1]
                    result_df.to_csv(result_path, index=False)
                    print(f"✅ Đã cập nhật id={result[0]} với response")
                else:
                    print(f"⚠️ Không có response cho id={result[0]}")
            # cập nhật tiến độ và in phần trăm
            async with progress_lock:
                progress_state['done'] += 1
                done = progress_state['done']
                percent = (done / total) * 100 if total else 100.0
                print(f"📈 Tiến độ: {done}/{total} ({percent:.2f}%)")
        finally:
            jobs_q.task_done()

async def run_all(rows, result_df, result_path: str, api_key, max_workers: int = None):
    total = len(rows)
    if max_workers is None:
        # Chạy tuần tự với 1 worker và 1 API key
        max_workers = 1

    jobs_q: asyncio.Queue = asyncio.Queue()
    for i, row in enumerate(rows):
        await jobs_q.put((i, row))
    for _ in range(max_workers):
        await jobs_q.put(None)

    df_lock = asyncio.Lock()
    progress_state = {'done': 0}
    progress_lock = asyncio.Lock()
    
    workers = [
        asyncio.create_task(
            worker(
                f"worker-{i+1}",
                api_key,
                jobs_q,
                result_df,
                result_path,
                total,
                df_lock,
                progress_state,
                progress_lock,
            )
        )
        for i in range(max_workers)
    ]
    await jobs_q.join()
    for w in workers:
        w.cancel()
    await asyncio.gather(*workers, return_exceptions=True)

# Cập nhật kết quả
print("Bắt đầu xử lý lại các bản ghi có response rỗng...")

# Chạy tuần tự với 1 worker để dùng lần lượt các key
asyncio.run(run_all(rows_to_process, result_df, result_path, api_key, max_workers=1))

print(f"✅ Hoàn tất xử lý lại. Kết quả cập nhật tại: {result_path}")

# Kiểm tra lại xem còn bản ghi nào rỗng không
final_check = result_df[result_df['response'].isna() | (result_df['response'] == '')]
if len(final_check) > 0:
    print(f"⚠️ Vẫn còn {len(final_check)} bản ghi có response rỗng: {final_check['ID'].tolist()}")
else:
    print("🎉 Tất cả bản ghi đã được xử lý thành công!")
