"""
ダミーデータCSV生成スクリプト
================================
実データに置き換えるまでの仮データとして以下のCSVを生成する。

生成ファイル:
  facility_master.csv - 施設マスタ（施設ID, 施設名, 都道府県, 地域ブロック）
  doctor_attr.csv     - 医師属性（医師ID, 施設ID, 年齢）
  activity_data.csv   - 活動データ（活動日, 施設ID, 医師ID, 品目ID, 活動種別）
                        ※ 1行 = 1活動イベント
  delivery_data.csv   - 納入データ（年月, 施設ID, 品目ID, 金額[百万円], 数量[千錠]）

実データに切り替える場合:
  - Databricks: spark.read.table(...).toPandas() の結果をCSV出力してこのファイルに置く
  - または load_data() 内の pd.read_csv() を直接 spark.read.table() に置き換える
"""

from __future__ import annotations

import calendar
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).parent
rng = np.random.default_rng(42)


# ============================================================
# 0. 施設マスタ (facility_master.csv)
# ============================================================

PREFECTURES = [
    ("北海道", "北海道・東北"),
    ("宮城県", "北海道・東北"),
    ("東京都", "関東"),
    ("神奈川県", "関東"),
    ("埼玉県", "関東"),
    ("千葉県", "関東"),
    ("愛知県", "中部"),
    ("静岡県", "中部"),
    ("大阪府", "近畿"),
    ("兵庫県", "近畿"),
    ("京都府", "近畿"),
    ("広島県", "中国・四国"),
    ("福岡県", "九州・沖縄"),
    ("熊本県", "九州・沖縄"),
]

# 都市部に施設が集中するよう重み付け
PREF_WEIGHTS = [3, 2, 12, 8, 5, 5, 7, 3, 9, 5, 4, 3, 5, 2]

N_FACILITIES = 300
pref_choices = rng.choice(
    len(PREFECTURES), size=N_FACILITIES,
    p=np.array(PREF_WEIGHTS) / sum(PREF_WEIGHTS)
)

facility_master = pd.DataFrame({
    "facility_id":   [f"F{i:04d}" for i in range(1, N_FACILITIES + 1)],
    "facility_name": [f"医療機関{i:04d}" for i in range(1, N_FACILITIES + 1)],
    "prefecture":    [PREFECTURES[p][0] for p in pref_choices],
    "region":        [PREFECTURES[p][1] for p in pref_choices],
})
facility_master.to_csv(BASE_DIR / "facility_master.csv", index=False, encoding="utf-8-sig")
print(f"[OK] facility_master.csv: {len(facility_master)} 行")

facilities = facility_master["facility_id"].tolist()


# ============================================================
# 1. 医師属性 (doctor_attr.csv)  ← facility_id 列を追加
# ============================================================

N_DOCTORS = 6000
doctor_attr = pd.DataFrame({
    "doctor_id":   [f"D{i:04d}" for i in range(1, N_DOCTORS + 1)],
    "facility_id": rng.choice(facilities, N_DOCTORS),
    "age":         rng.integers(33, 65, N_DOCTORS),
})
doctor_attr.to_csv(BASE_DIR / "doctor_attr.csv", index=False, encoding="utf-8-sig")
print(f"[OK] doctor_attr.csv: {len(doctor_attr)} 行")

# doctor_id → facility_id の辞書
doc_to_facility: dict[str, str] = dict(
    zip(doctor_attr["doctor_id"], doctor_attr["facility_id"])
)


# ============================================================
# 2. 活動データ (activity_data.csv)
#    1行 = 1活動イベント
#    列: activity_date, facility_id, doctor_id, product_id, activity_type
# ============================================================

def get_fy_months(fy_start: int) -> list[str]:
    return (
        [f"{fy_start}-{m:02d}" for m in range(4, 13)]
        + [f"{fy_start + 1}-{m:02d}" for m in range(1, 4)]
    )

# 実績データは 2026-03-31 まで（FY2024 + FY2025 の2年分）
months_hist = get_fy_months(2024) + get_fy_months(2025)

# 品目別目標頻度（回/医師/月）
target_freq: dict[str, float] = {
    "GLI": 1.4, "CUV": 1.3, "HYQ": 1.2,
    "INT": 1.5, "TRI": 1.6, "ENT": 1.6,
    "OVE": 1.4,
    "LIV": 1.2, "REV": 1.1, "ALC": 1.0,
    "VYV": 1.3, "VPR": 1.3,
    "LVM": 1.0,
}

# ターゲット医師範囲
doctor_ranges: dict[str, tuple[int, int]] = {
    "GLI": (1,    1400), "CUV": (1001, 2100), "HYQ": (1200, 1949),
    "INT": (1,    1800), "TRI": (931,  2580), "ENT": (801,  2450),
    "OVE": (1,     600),
    "LIV": (2001, 3000), "REV": (2301, 3200), "ALC": (2501, 3200),
    "VYV": (3001, 3800), "VPR": (3201, 3940),
    "LVM": (4001, 4400),
}

# 活動種別とその出現割合（全品目共通ベース）
# Web系・メールは品目ごとに調整可能
ACTIVITY_TYPES  = ["面談", "面談_アポ", "説明会", "Web講演会", "メール", "Web面談"]
ACTIVITY_WEIGHTS = [0.32,   0.18,      0.06,    0.05,       0.26,    0.13]

act_records: list[dict] = []

for pid, (start, end) in doctor_ranges.items():
    docs = [f"D{i:04d}" for i in range(start, end + 1)]
    freq = target_freq[pid]
    p2 = max(0.0, min(1.0, freq - 1.0))
    p1 = 1.0 - p2

    for ym in months_hist:
        year, month = map(int, ym.split("-"))
        days_in_month = calendar.monthrange(year, month)[1]

        for doc in docs:
            if rng.random() > 0.85:
                continue  # 接触なし
            n_contacts = int(rng.choice([1, 2], p=[p1, p2]))
            for _ in range(n_contacts):
                day = int(rng.integers(1, days_in_month + 1))
                act_type = rng.choice(ACTIVITY_TYPES, p=ACTIVITY_WEIGHTS)
                act_records.append({
                    "activity_date": f"{year}-{month:02d}-{day:02d}",
                    "facility_id":   doc_to_facility[doc],
                    "doctor_id":     doc,
                    "product_id":    pid,
                    "activity_type": act_type,
                })

activity_data = pd.DataFrame(act_records)
activity_data.to_csv(BASE_DIR / "activity_data.csv", index=False, encoding="utf-8-sig")
print(f"[OK] activity_data.csv: {len(activity_data):,} 行")


# ============================================================
# 3. 納入データ (delivery_data.csv)  ← facility_id 列を追加
#    列: delivery_ym, facility_id, product_id, amount_mn, quantity_k
# ============================================================

sales_forecast = pd.read_csv(BASE_DIR / "sales_forecast.csv")

# 納入実績データも 2026-03-31 まで（活動データと同期間）
all_months = get_fy_months(2024) + get_fy_months(2025)

def fy_of(ym: str) -> str:
    dt_year, dt_month = map(int, ym.split("-"))
    fy_year = dt_year if dt_month >= 4 else dt_year - 1
    return f"FY{fy_year}"

unit_price: dict[str, float] = {
    "GLI": 1.2, "CUV": 1.5, "HYQ": 1.3,
    "INT": 0.9, "TRI": 0.8, "ENT": 1.0,
    "OVE": 1.8,
    "LIV": 1.1, "REV": 1.3, "ALC": 1.0,
    "VYV": 1.6, "VPR": 1.4,
    "LVM": 5.0,
    "Zaso": 2.0, "WSA": 1.9,
    "TKZ": 4.5, "RPL": 4.0, "VON": 4.2,
}

# FY2026 を基準に FY2024/FY2025 の実績金額を逆算
# 現実世界のモデル: FY2025 ≈ FY2026 × 1.05、FY2024 ≈ FY2026 × 1.10
# 未発売品（launch_ym > 実績期間）は 0
products_df = pd.read_csv(BASE_DIR / "products.csv", dtype=str)
launch_map: dict[str, str] = dict(zip(products_df["product_id"], products_df["launch_ym"]))

def get_hist_annual_mn(row: pd.Series, fy: str) -> float:
    """FY2024/FY2025 の年間金額を FY2026 ベースで推計する。"""
    pid = row["product_id"]
    base = float(row.get("FY2026", 0) or 0)
    if base <= 0:
        return 0.0
    launch_ym = launch_map.get(pid, "2000-01")
    fy_year = int(fy[2:])
    launch_year = int(launch_ym[:4])
    launch_month = int(launch_ym[5:7])
    # 発売FY: 発売月が4月以降ならその年、そうでなければ前年
    launch_fy = launch_year if launch_month >= 4 else launch_year - 1
    if fy_year < launch_fy:
        return 0.0  # 未発売
    # 成長率（逆向き）: FY2025 = FY2026 × 1.05、FY2024 = FY2026 × 1.10
    growth = {2025: 1.05, 2024: 1.10}.get(fy_year, 1.0)
    return base * growth

# 品目ごとに担当施設をターゲット医師の施設から特定する
# product → set of facility_ids
product_facilities: dict[str, list[str]] = {}
for pid, (start, end) in doctor_ranges.items():
    docs = [f"D{i:04d}" for i in range(start, end + 1)]
    facs = list({doc_to_facility[d] for d in docs})
    product_facilities[pid] = facs

del_records: list[dict] = []
for _, row in sales_forecast.iterrows():
    pid = row["product_id"]
    fac_list = product_facilities.get(pid, facilities)
    n_fac = len(fac_list)

    for ym in all_months:
        fy = fy_of(ym)
        annual_mn = get_hist_annual_mn(row, fy)
        if annual_mn <= 0:
            continue

        # 施設間でランダムウェイトで按分（ディリクレ分布）
        fac_weights = rng.dirichlet(np.ones(n_fac))
        monthly_total = annual_mn / 12.0 * rng.uniform(0.90, 1.10)
        up = unit_price.get(pid, 1.0)

        for fac_id, w in zip(fac_list, fac_weights):
            amount = round(monthly_total * w, 3)
            if amount < 0.001:
                continue
            del_records.append({
                "delivery_ym":  ym,
                "facility_id":  fac_id,
                "product_id":   pid,
                "amount_mn":    amount,
                "quantity_k":   round(amount / up, 3),
            })

delivery_data = pd.DataFrame(del_records)
delivery_data.to_csv(BASE_DIR / "delivery_data.csv", index=False, encoding="utf-8-sig")
print(f"[OK] delivery_data.csv: {len(delivery_data):,} 行")

print("\nすべてのデータCSVを生成しました。")
print("実データに切り替える場合はこれらのCSVを実データで上書きしてください。")
