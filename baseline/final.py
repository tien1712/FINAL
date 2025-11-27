import os, json, re, time, ast
import asyncio
from collections import deque
from datetime import datetime, timezone, timedelta
import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm
from langchain_google_genai import ChatGoogleGenerativeAI
from prompt import prompt
import csv
# code xử lý tiếp các bản ghi có prediction rỗng
load_dotenv()

# Lấy tối đa 11 API keys từ biến môi trường GOOGLE_API_KEY_1..11 (lọc None/rỗng)
raw_api_keys = [os.getenv(f"GOOGLE_API_KEY_{i}") for i in range(1, 2)]
api_keys = [k for k in raw_api_keys if k and str(k).strip()]
if not api_keys:
    raise RuntimeError("Không tìm thấy API key nào trong biến môi trường GOOGLE_API_KEY_1")

label_map = {
    "public transports": 0,
    "private modes": 1,
    "soft modes": 2
}

# Đọc dữ liệu gốc
df = pd.read_csv("data/test.csv")
df["id"] = df.index  # lưu lại chỉ số dòng gốc

# Đọc/khởi tạo kết quả hiện tại
result_path = "results/result1.csv"
process_all = False
try:
    result_df = pd.read_csv(result_path)
    if 'id' not in result_df.columns or 'prediction' not in result_df.columns:
        raise ValueError("result1.csv không đúng định dạng")
    # Tìm các bản ghi có prediction rỗng
    empty_predictions = result_df[result_df['prediction'].isna() | (result_df['prediction'] == '')]
    print(f"Tìm thấy {len(empty_predictions)} bản ghi có prediction rỗng")
    if len(empty_predictions) == 0:
        # Nếu không còn rỗng, không cần xử lý thêm
        print("Không có bản ghi nào cần xử lý lại!")
        # Không exit để có thể hỗ trợ chạy toàn tập nếu người dùng xóa file rồi chạy lại
        rows_to_process = []
    else:
        empty_ids = empty_predictions['id'].tolist()
        rows_to_process = df[df['id'].isin(empty_ids)].to_dict(orient="records")
        print(f"Sẽ xử lý lại {len(rows_to_process)} bản ghi")
except Exception:
    # File chưa tồn tại hoặc không hợp lệ → khởi tạo mới và xử lý toàn bộ test
    os.makedirs(os.path.dirname(result_path), exist_ok=True)
    base = {
        'id': df['id'],
        'prediction': [''] * len(df)
    }
    if 'CHOICE' in df.columns:
        base['CHOICE'] = df['CHOICE']
    else:
        base['CHOICE'] = [None] * len(df)
    result_df = pd.DataFrame(base)
    result_df.to_csv(result_path, index=False)
    rows_to_process = df.to_dict(orient="records")
    process_all = True
    print(f"Khởi tạo {result_path}. Sẽ xử lý toàn bộ {len(rows_to_process)} bản ghi trong test.csv")

def safe_parse_json(raw: str):
    if raw is None or not str(raw).strip():
        raise ValueError("Empty model response")
    text = str(raw).strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE | re.MULTILINE)
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
        # Heuristic cuối: cố gắng trích xuất nhãn dự đoán từ text tự do
        try:
            # Tìm pattern prediction: <label> hoặc chỉ có label đứng riêng lẻ
            m = re.search(r"prediction\s*[:=\-\s]+(Public transports|Private modes|Soft modes)", text, flags=re.IGNORECASE)
            if not m:
                # Tìm label đứng riêng lẻ (có thể là toàn bộ response)
                m = re.search(r"\b(Public transports|Private modes|Soft modes)\b", text, flags=re.IGNORECASE)
            if m:
                label = m.group(1).strip()
                return {"prediction": label}
        except Exception:
            pass
        # Nếu tất cả đều thất bại, ném lỗi gốc để hiển thị thông tin
        raise

class PerKeyRateLimiter:
    def __init__(self, key: str, per_minute_limit: int = 2, per_day_limit: int = 50):
        self.key = key
        self.per_minute_limit = per_minute_limit
        self.per_day_limit = per_day_limit
        self.minute_window = deque()  # timestamps giây trong 60s gần nhất
        self.day_count = 0
        self.day_anchor = self._current_day_anchor()
        self.lock = asyncio.Lock()

    def _current_day_anchor(self) -> str:
        return datetime.now(timezone.utc).date().isoformat()

    def _reset_day_if_needed(self):
        current = self._current_day_anchor()
        if current != self.day_anchor:
            self.day_anchor = current
            self.day_count = 0

    def _prune_minute_window(self, now_ts: float):
        cutoff = now_ts - 60.0
        while self.minute_window and self.minute_window[0] <= cutoff:
            self.minute_window.popleft()

    async def wait_for_slot(self):
        while True:
            async with self.lock:
                now_ts = time.time()
                self._reset_day_if_needed()
                self._prune_minute_window(now_ts)

                minute_used = len(self.minute_window)
                day_used = self.day_count

                if day_used >= self.per_day_limit:
                    # ngủ đến đầu ngày UTC tiếp theo
                    now_dt = datetime.now(timezone.utc)
                    next_day = (now_dt + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
                    sleep_s = max(1.0, (next_day - now_dt).total_seconds())
                elif minute_used < self.per_minute_limit:
                    self.minute_window.append(now_ts)
                    self.day_count += 1
                    return
                else:
                    oldest = self.minute_window[0]
                    sleep_s = max(0.0, 60.0 - (now_ts - oldest)) + 0.01
            await asyncio.sleep(sleep_s)

class ApiKeyPool:
    def __init__(self, api_keys_list, per_minute_limit: int = 2, per_day_limit: int = 50):
        self.limiters = [PerKeyRateLimiter(k, per_minute_limit, per_day_limit) for k in api_keys_list]
        self.idx = 0
        self.pool_lock = asyncio.Lock()

    async def acquire(self) -> PerKeyRateLimiter:
        while True:
            # xoay vòng key để tránh starvation
            async with self.pool_lock:
                start = self.idx
                order = list(range(len(self.limiters)))
                order = order[start:] + order[:start]

            # thử lấy slot ngay
            for i in order:
                limiter = self.limiters[i]
                if await self._try_immediate(limiter):
                    async with self.pool_lock:
                        self.idx = (i + 1) % len(self.limiters)
                    return limiter

            # nếu không có slot ngay, chờ ngắn rồi thử lại
            await asyncio.sleep(0.05)

    async def _try_immediate(self, limiter: PerKeyRateLimiter) -> bool:
        now_ts = time.time()
        async with limiter.lock:
            limiter._reset_day_if_needed()
            limiter._prune_minute_window(now_ts)
            if limiter.day_count >= limiter.per_day_limit:
                return False
            if len(limiter.minute_window) < limiter.per_minute_limit:
                limiter.minute_window.append(now_ts)
                limiter.day_count += 1
                return True
            return False

async def call_model_async(index, total, row, api_key, retries=2, delay=30):
    # Tạo model theo key mỗi lần gọi để tránh xung đột state giữa threads
    def _do_invoke(prompt_text: str):
        model = ChatGoogleGenerativeAI(model="gemini-2.5-pro", google_api_key=api_key)
        return model.invoke(prompt_text)

    for attempt in range(retries):
        try:
            prompt_text = prompt(row)
            print(f"[{index+1}/{total}] 🔍 Đang xử lý lại row id={row['id']}...")
            response = await asyncio.to_thread(_do_invoke, prompt_text)
            raw = str(getattr(response, "content", response))

            try:
                prediction_dict = safe_parse_json(raw)
            except Exception as pe:
                print(f"[{index+1}] ❌ JSON parse failed for id={row.get('id','N/A')}: {pe}")
                with open("errors.log", "a") as f:
                    f.write(f"JSON parse failed at index {index} (id={row.get('id','N/A')}): {pe}\nRAW:\n{raw}\n\n")
                return row.get("id", index), None, row.get("CHOICE", None)

            prediction_str = (prediction_dict.get("prediction") or "").strip().lower()
            print(f"[{index+1}] ✅ Hoàn tất row id={row['id']} → {prediction_str}")
            return row["id"], label_map.get(prediction_str, None), row.get("CHOICE", None)
        except Exception as e:
            if ("429" in str(e) or "rate" in str(e).lower()) and attempt < retries - 1:
                print(f"[{index+1}] ⏳ Lỗi 429/rate limit. Đợi {delay}s rồi thử lại...")
                await asyncio.sleep(delay)
            else:
                print(f"[{index+1}] ❌ Lỗi ở row id={row.get('id', 'N/A')}: {e}")
                with open("errors.log", "a") as f:
                    f.write(f"Lỗi ở dòng {index} (id={row.get('id', 'N/A')}): {e}\n")
                return row.get("id", index), None, row.get("CHOICE", None)

async def process_with_single_key(index, total, row, api_key, retries=5, delay=30):
    """Xử lý 1 request với 1 key, retry tối đa 5 lần nếu lỗi"""
    print(f"[{index+1}] 🔑 Dùng API key, thử tối đa {retries} lần...")
    result = await call_model_async(index, total, row, api_key, retries=retries, delay=delay)
    return result

async def worker(name, api_keys_list, jobs_q: asyncio.Queue, result_df, result_path: str, total: int, df_lock: asyncio.Lock, progress_state: dict, progress_lock: asyncio.Lock, key_index_state: dict, key_lock: asyncio.Lock):
    while True:
        item = await jobs_q.get()
        if item is None:
            jobs_q.task_done()
            break

        try:
            # Lấy key theo round-robin
            async with key_lock:
                current_key_index = key_index_state['index']
                api_key = api_keys_list[current_key_index]
                key_index_state['index'] = (current_key_index + 1) % len(api_keys_list)
                print(f"[{item[0]+1}] 🔄 Sử dụng key {current_key_index + 1}/{len(api_keys_list)}")
            
            # Xử lý request với 1 key, retry 5 lần nếu lỗi
            result = await process_with_single_key(item[0], total, item[1], api_key, retries=5, delay=30)
            # cập nhật kết quả và lưu file an toàn
            async with df_lock:
                result_df.loc[result_df['id'] == result[0], 'prediction'] = result[1]
                # Cập nhật CHOICE nếu còn trống và có trong dữ liệu đầu vào
                if 'CHOICE' in result_df.columns and 'CHOICE' in item[1]:
                    if pd.isna(result_df.loc[result_df['id'] == result[0], 'CHOICE']).any() or (result_df.loc[result_df['id'] == result[0], 'CHOICE'] == '').any():
                        result_df.loc[result_df['id'] == result[0], 'CHOICE'] = item[1].get('CHOICE')
                result_df.to_csv(result_path, index=False)
                print(f"✅ Đã cập nhật id={result[0]} với prediction={result[1]}")
            # cập nhật tiến độ và in phần trăm
            async with progress_lock:
                progress_state['done'] += 1
                done = progress_state['done']
                percent = (done / total) * 100 if total else 100.0
                print(f"📈 Tiến độ: {done}/{total} ({percent:.2f}%)")
        finally:
            jobs_q.task_done()

async def run_all(rows, result_df, result_path: str, api_keys_list, max_workers: int = None):
    total = len(rows)
    if max_workers is None:
        # Chạy tuần tự với 1 worker để dùng lần lượt các key
        max_workers = 1

    jobs_q: asyncio.Queue = asyncio.Queue()
    for i, row in enumerate(rows):
        await jobs_q.put((i, row))
    for _ in range(max_workers):
        await jobs_q.put(None)

    df_lock = asyncio.Lock()
    progress_state = {'done': 0}
    progress_lock = asyncio.Lock()
    key_index_state = {'index': 0}  # State để track key hiện tại
    key_lock = asyncio.Lock()  # Lock để đảm bảo thread-safe khi lấy key
    
    workers = [
        asyncio.create_task(
            worker(
                f"worker-{i+1}",
                api_keys_list,
                jobs_q,
                result_df,
                result_path,
                total,
                df_lock,
                progress_state,
                progress_lock,
                key_index_state,
                key_lock,
            )
        )
        for i in range(max_workers)
    ]
    await jobs_q.join()
    for w in workers:
        w.cancel()
    await asyncio.gather(*workers, return_exceptions=True)

# Cập nhật kết quả
print("Bắt đầu xử lý lại các bản ghi có prediction rỗng...")

# Chạy tuần tự với 1 worker để dùng lần lượt các key
asyncio.run(run_all(rows_to_process, result_df, result_path, api_keys, max_workers=1))

print(f"✅ Hoàn tất xử lý lại. Kết quả cập nhật tại: {result_path}")

# Kiểm tra lại xem còn bản ghi nào rỗng không
final_check = result_df[result_df['prediction'].isna() | (result_df['prediction'] == '')]
if len(final_check) > 0:
    print(f"⚠️ Vẫn còn {len(final_check)} bản ghi có prediction rỗng: {final_check['id'].tolist()}")
else:
    print("🎉 Tất cả bản ghi đã được xử lý thành công!")
