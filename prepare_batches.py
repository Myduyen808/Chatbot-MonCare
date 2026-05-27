# prepare_batches.py, Chia KB1 (400 câu) thành 4 batch, mỗi batch 100 câu:
import pandas as pd

df = pd.read_excel("KB3_Information_Noise.xlsx")
questions = df["Câu hỏi người dùng (Input)"].dropna().tolist()
answers   = df["Phản hồi kỳ vọng (Expected Output)"].dropna().tolist()

for i in range(4):
    start = i * 100
    end   = start + 100
    batch = pd.DataFrame({
        "question":         questions[start:end],
        "ground_truth":     answers[start:end],
    })
    batch.to_csv(f"kb3_batch_{i+1}.csv", index=False, encoding="utf-8-sig")
    print(f"Batch {i+1}: {len(batch)} câu → kb3_batch_{i+1}.csv")