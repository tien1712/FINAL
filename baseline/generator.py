import pandas as pd
from sklearn.model_selection import train_test_split

# Đọc dữ liệu từ file
df = pd.read_csv("original_data/textdata.csv")  # Đảm bảo file nằm cùng thư mục với script

# Chia dữ liệu theo tỉ lệ 70:30 (train:test)
train_df, test_df = train_test_split(df, test_size=0.3, stratify=df['CHOICE'])

# Lấy các behavior cần thiết
a = test_df["ID"].unique()
df1 = pd.read_csv("original_data/behavior.csv")
df2 = df1[df1["ID"].isin(a)]


# Lưu ra file
train_df.to_csv("data/train.csv", index=False)
test_df.to_csv("data/test.csv", index=False)
df2.to_csv("data/behavior.csv", index=False)

# In thông tin về kích thước các tập dữ liệu
print(f"Tổng số mẫu: {len(df)}")
print(f"Train set: {len(train_df)} mẫu ({len(train_df)/len(df)*100:.1f}%)")
print(f"Test set: {len(test_df)} mẫu ({len(test_df)/len(df)*100:.1f}%)")
