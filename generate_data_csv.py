"""
ダミーデータCSV生成スクリプト
================================
実データに置き換えるまでの仮データとして以下のCSVを生成する。

生成ファイル:
  activity_data.csv  - 活動データ（品目×医師×月別コール数）
  doctor_attr.csv    - 医師属性（医師ID, 年齢）
  delivery_data.csv  - 納入データ（品目×月別金額・数量）

実データに切り替える場合:
  - Databricks: spark.read.table(...).toPandas() の結果をCSV出力してこのファイルに置く
  - または load_data() 内の pd.read_csv() を直接 spark.read.table() に置き換える
"""

from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).parent
np.random.seed(42)


# ============================================================
# 1. 医師属性 (doctor_attr.csv)
# ============================================================

N_DOCTORS = 6000
doctor_attr = pd.DataFrame({
    "doctor_id": [f"D{i:04d}" for i in range(1, N_DOCTORS + 1)],
    "age":       np.random.randint(33, 65, N_DOCTORS),
})
doctor_attr.to_csv(BASE_DIR / "doctor_attr.csv", index=False, encoding="utf-8-sig")
print(f"[OK] doctor_attr.csv: {len(doctor_attr)} 行")


# ============================================================
# 2. 活動データ (activity_data.csv)
# ============================================================
# FY2028 の12ヶ月（頻度推定の基準期間）
# 品目ごとの目標頻度に基づいて活動回数をサンプリング

def get_fy_months(fy_start: int):
    return (
        [f"{fy_start}-{m:02d}" for m in range(4, 13)]
        + [f"{fy_start + 1}-{m:02d}" for m in range(1, 4)]
    )

months_hist = get_fy_months(2028)

# 品目別目標頻度（回/医師/月）
target_freq = {
    "GLI": 1.4, "CUV": 1.3, "HYQ": 1.2,
    "INT": 1.5, "TRI": 1.6, "ENT": 1.6,
    "OVE": 1.4,
    "LIV": 1.2, "REV": 1.1, "ALC": 1.0,
    "VYV": 1.3, "VPR": 1.3,
    "LVM": 1.0,
}

# ターゲット医師範囲（target_doctor_ranges.csv と同じ）
doctor_ranges = {
    "GLI": (1,    1400), "CUV": (1001, 2100), "HYQ": (1200, 1949),
    "INT": (1,    1800), "TRI": (931,  2580), "ENT": (801,  2450),
    "OVE": (1,     600),
    "LIV": (2001, 3000), "REV": (2301, 3200), "ALC": (2501, 3200),
    "VYV": (3001, 3800), "VPR": (3201, 3940),
    "LVM": (4001, 4400),
}

records = []
for pid, (start, end) in doctor_ranges.items():
    docs = [f"D{i:04d}" for i in range(start, end + 1)]
    freq = target_freq[pid]
    p2 = max(0.0, min(1.0, freq - 1.0))
    p1 = 1.0 - p2
    for ym in months_hist:
        for doc in docs:
            if np.random.rand() < 0.85:   # 85%の確率で接触あり
                records.append({
                    "activity_ym":    ym,
                    "product_id":     pid,
                    "doctor_id":      doc,
                    "activity_count": int(np.random.choice([1, 2], p=[p1, p2])),
                })

activity_data = pd.DataFrame(records)
activity_data.to_csv(BASE_DIR / "activity_data.csv", index=False, encoding="utf-8-sig")
print(f"[OK] activity_data.csv: {len(activity_data):,} 行")


# ============================================================
# 3. 納入データ (delivery_data.csv)
# ============================================================
# FY2026〜FY2029 の月次納入量（金額: 百万円, 数量: 千錠）
# 年次予測 sales_forecast.csv の値を月次に按分（均等 + ±10%ノイズ）

sales_forecast = pd.read_csv(BASE_DIR / "sales_forecast.csv")
all_months = get_fy_months(2026) + get_fy_months(2027) + get_fy_months(2028) + get_fy_months(2029)

def fy_of(ym: str) -> str:
    from datetime import datetime
    dt = datetime.strptime(ym, "%Y-%m")
    return f"FY{dt.year if dt.month >= 4 else dt.year - 1}"

# 単価（千錠あたり百万円）の仮定
unit_price = {
    "GLI": 1.2, "CUV": 1.5, "HYQ": 1.3,
    "INT": 0.9, "TRI": 0.8, "ENT": 1.0,
    "OVE": 1.8,
    "LIV": 1.1, "REV": 1.3, "ALC": 1.0,
    "VYV": 1.6, "VPR": 1.4,
    "LVM": 5.0,  # 希少疾患は高単価
}

del_records = []
for _, row in sales_forecast.iterrows():
    pid = row["product_id"]
    for ym in all_months:
        fy = fy_of(ym)
        annual_mn = float(row.get(fy, 0) or 0)
        if annual_mn <= 0:
            del_records.append({"delivery_ym": ym, "product_id": pid,
                                 "amount_mn": 0.0, "quantity_k": 0.0})
            continue
        monthly_base = annual_mn / 12.0
        noise = np.random.uniform(0.90, 1.10)
        amount = round(monthly_base * noise, 1)
        up = unit_price.get(pid, 1.0)
        quantity = round(amount / up, 1)
        del_records.append({
            "delivery_ym": ym,
            "product_id":  pid,
            "amount_mn":   amount,    # 百万円
            "quantity_k":  quantity,  # 千錠
        })

delivery_data = pd.DataFrame(del_records)
delivery_data.to_csv(BASE_DIR / "delivery_data.csv", index=False, encoding="utf-8-sig")
print(f"[OK] delivery_data.csv: {len(delivery_data):,} 行")


print("\nすべてのデータCSVを生成しました。")
print("実データに切り替える場合はこれらのCSVを実データで上書きしてください。")
