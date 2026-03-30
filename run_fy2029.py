
# %%
"""
FTE算出 メイン実行スクリプト（全品目・FY2026〜2035対応）
=================================================================
対象品目（13品目）:
  CS: GLI, CUV, HYQ, INT, TRI, OVE, ENT, LIV, REV, ALC, VYV, VPR
  PS: LVM（希少疾患）

対象期間: FY2026〜FY2035（120ヶ月）
OVE発売: FY2026（2026年7月想定）
TRI LOE: FY2031（2031-07）
ENT LOE: FY2032（2032-10）

出力ファイル:
  output/fy2029_fte_report.html   - メインレポート（FYトレンド含む）
  output/fy2029_sensitivity.html  - 感度分析
  output/fy2029_fte_detail.csv    - 詳細データ
  output/fy2029_summary.csv       - 年度サマリー
  output/fy2029_allocation.csv    - OVE FTE配分
"""

from __future__ import annotations
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from fy2029_fte_calculator import (
    ProductConfig,
    TargetDoctorCalculator,
    FCScAllocator,
    ActivityFrequencyEstimator,
    MRDigitalRatioEstimator,
    NewProductFTEAllocator,
    FY2029FTECalculator,
    get_fy_months,
    normalize_fte_to_headcount,
    discretize_fte_semiannually,
    calculate_roi_optimal_fte,
    CURRENT_MR_COUNT,
)
from fy2029_html_reporter import FY2029HTMLReporter
from fy2029_sensitivity import FY2029SensitivityAnalyzer

OUTPUT_DIR = Path(__file__).parent / "output"
PRODUCTS_CSV = Path(__file__).parent / "products.csv"

# FY2026〜FY2035 の120ヶ月
ALL_MONTHS = sum([get_fy_months(fy) for fy in range(2026, 2036)], [])


def normalize_ym(value: str) -> str:
    clean = str(value).strip()
    if not clean or clean.lower() in ("nan", "none"):
        raise ValueError(f"Invalid year-month value: '{value}'")
    for fmt in ("%Y-%m", "%y-%m", "%b-%y", "%b-%Y", "%Y/%m", "%Y.%m"):
        try:
            dt = datetime.strptime(clean, fmt)
            if fmt in ("%y-%m",):
                if dt.year < 1900:
                    dt = dt.replace(year=dt.year + 100)
            return dt.strftime("%Y-%m")
        except ValueError:
            continue
    raise ValueError(f"Invalid year-month format: '{value}'")


# ============================================================
# 品目設定CSV読み込み
# ============================================================

def load_product_configs(
    csv_path: Path = PRODUCTS_CSV,
) -> Tuple[List[ProductConfig], Dict[str, str]]:
    """
    products.csv から品目設定を読み込む。

    CSVカラム:
      product_id, area, is_new, launch_ym, loe_ym,
      estimated_patients, num_indications, reference_product

    loe_months は loe_ym と launch_ym の差分から自動計算する。
    reference_product が空でない品目は新発売品として参照先を登録。

    Returns
    -------
    (configs, reference_products)
    """
    df = pd.read_csv(csv_path, dtype=str)
    configs: List[ProductConfig] = []
    reference_products: Dict[str, str] = {}

    for _, row in df.iterrows():
        launch = datetime.strptime(row["launch_ym"].strip(), "%Y-%m")
        loe    = datetime.strptime(row["loe_ym"].strip(),    "%Y-%m")
        loe_months = (loe.year - launch.year) * 12 + (loe.month - launch.month)

        # 効能追加パラメータ（CSVに列がない場合 or NaN の場合はデフォルト）
        ind_ym_raw = row.get("indication_add_ym", "")
        ind_ym = "" if pd.isna(ind_ym_raw) else str(ind_ym_raw).strip()
        ind_boost_raw = row.get("indication_fte_boost", 1.0)
        ind_boost = 1.0 if (pd.isna(ind_boost_raw) or str(ind_boost_raw).strip() == "") else float(ind_boost_raw)
        ind_months_raw = row.get("indication_boost_months", 0)
        ind_months = 0 if (pd.isna(ind_months_raw) or str(ind_months_raw).strip() == "") else int(float(ind_months_raw))

        # バイオシミラー耐性パラメータ（CSVに列がない場合はデフォルト 0.0）
        post_loe_raw = row.get("post_loe_factor", 0.0)
        post_loe = 0.0 if (pd.isna(post_loe_raw) or str(post_loe_raw).strip() == "") else float(post_loe_raw)

        configs.append(ProductConfig(
            product_id            = row["product_id"].strip(),
            area                  = row["area"].strip(),
            is_new                = row["is_new"].strip().lower() == "true",
            launch_ym             = row["launch_ym"].strip(),
            loe_months            = loe_months,
            estimated_patients    = int(row["estimated_patients"]),
            num_indications       = int(row["num_indications"]),
            indication_add_ym     = ind_ym if ind_ym and ind_ym.lower() not in ("", "nan", "none") else None,
            indication_fte_boost  = ind_boost,
            indication_boost_months = ind_months,
            post_loe_factor       = post_loe,
        ))

        ref = str(row.get("reference_product", "")).strip()
        if ref and ref.lower() not in ("", "nan", "none"):
            reference_products[row["product_id"].strip()] = ref

    return configs, reference_products


# ============================================================
# データ読み込み
# ============================================================

def load_data():
    """
    全データをCSVから読み込んで返す。

    読み込むCSVファイル（同ディレクトリ）:
      activity_data.csv        - 活動データ（generate_data_csv.py で生成）
      doctor_attr.csv          - 医師属性（generate_data_csv.py で生成）
      delivery_data.csv        - 納入データ（generate_data_csv.py で生成）
      sales_forecast.csv       - 売上予測（手動編集可）
      mmm_decay_params.csv     - MMM減衰パラメータ（手動編集可）
      activity_set.csv         - FC/SC品目セット（手動編集可）
      target_doctors.csv       - 品目別ターゲット医師数（手動編集可）
      target_doctor_ranges.csv - 医師IDレンジ（被り率計算用、手動編集可）
      current_activities.csv   - 現在のMR/Digital活動量（手動編集可）
      current_fte.csv          - 現在のFTE（手動編集可）

    Databricks本番環境に切り替える場合:
      pd.read_csv(...) を spark.read.table(...).toPandas() に置き換える
    """
    BASE = Path(__file__).parent

    # ---- 活動データ ----
    # 新スキーマ: activity_date, facility_id, doctor_id, product_id, activity_type（1行=1活動）
    # FTE計算モジュールが期待する旧スキーマ（activity_ym, doctor_id, product_id, activity_count）に変換
    _act_raw = pd.read_csv(BASE / "activity_data.csv")
    if "activity_date" in _act_raw.columns and "activity_ym" not in _act_raw.columns:
        _act_raw["activity_ym"] = _act_raw["activity_date"].str[:7]
        activity_data = (
            _act_raw.groupby(["activity_ym", "product_id", "doctor_id"])
            .size()
            .reset_index(name="activity_count")
        )
    else:
        activity_data = _act_raw

    # ---- 医師属性 ----
    doctor_attr = pd.read_csv(BASE / "doctor_attr.csv")

    # ---- MMM減衰パラメータ ----
    decay_params_df = pd.read_csv(BASE / "mmm_decay_params.csv")

    # ---- FC/SC活動セット ----
    activity_set_df = pd.read_csv(BASE / "activity_set.csv")

    # ---- ターゲット医師IDリスト（被り率計算用）----
    # target_doctor_ranges.csv の start〜end から pd.Series を生成
    ranges_df = pd.read_csv(BASE / "target_doctor_ranges.csv")
    target_doctor_lists = {
        row["product_id"]: pd.Series([
            f"D{i:04d}"
            for i in range(int(row["doctor_id_start"]), int(row["doctor_id_end"]) + 1)
        ])
        for _, row in ranges_df.iterrows()
    }

    # ---- FC/SC 比率（明示指定）----
    fc_ratio_df = pd.read_csv(BASE / "fc_sc_ratio.csv")
    fc_ratios = dict(zip(fc_ratio_df["product_id"], fc_ratio_df["fc_ratio"].astype(float)))

    # ---- 現在の活動量 ----
    ca_df = pd.read_csv(BASE / "current_activities.csv")
    current_activities = {
        row["product_id"]: {"MR": float(row["mr_activity"]), "Digital": float(row["digital_activity"])}
        for _, row in ca_df.iterrows()
    }

    # ---- 現在のFTE ----
    fte_df = pd.read_csv(BASE / "current_fte.csv")
    current_fte = dict(zip(fte_df["product_id"], fte_df["current_fte"].astype(float)))

    # ---- 売上予測・納入データ（将来分析用として保持）----
    sales_forecast = pd.read_csv(BASE / "sales_forecast.csv")
    delivery_data  = pd.read_csv(BASE / "delivery_data.csv")

    # ---- 供給制限スケジュール（supply_restriction.csv）----
    supply_path = BASE / "supply_restriction.csv"
    supply_restrictions: Dict[str, list] = {}
    if supply_path.exists():
        sr_df = pd.read_csv(supply_path, dtype=str)
        for _, row in sr_df.iterrows():
            pid = row["product_id"].strip()
            supply_restrictions.setdefault(pid, []).append({
                "start_ym": normalize_ym(row["restriction_start_ym"]),
                "end_ym":   normalize_ym(row["restriction_end_ym"]),
                "factor":   float(row["restriction_factor"]),
            })
        n = sum(len(v) for v in supply_restrictions.values())
        print(f"       → 供給制限: {n} 件読み込み ({', '.join(supply_restrictions.keys())})")

    return dict(
        activity_data        = activity_data,
        doctor_attr          = doctor_attr,
        supply_restrictions  = supply_restrictions,
        product_info       = None,          # products.csv で管理（main()で上書き）
        decay_params_df    = decay_params_df,
        activity_set_df    = activity_set_df,
        target_doctor_lists= target_doctor_lists,
        current_activities = current_activities,
        current_fte        = current_fte,
        sales_forecast     = sales_forecast,
        delivery_data      = delivery_data,
        fc_ratios          = fc_ratios,
        mmm_optimal_df     = None,
    )


# ============================================================
# メイン処理
# ============================================================

def main():
    print("=" * 65)
    print("FY2029 FTE算出（全品目・FY2026〜FY2035）開始")
    print("=" * 65)

    # ---- 1. データ読み込み ----
    print("\n[1/6] データ読み込み...")
    data = load_data()

    # ---- 2. 品目設定（products.csv から読み込み）+ MR比率パラメータ ----
    print(f"[2/6] 品目設定（{PRODUCTS_CSV.name} から読み込み）...")
    product_configs, csv_reference_products = load_product_configs(PRODUCTS_CSV)

    # mr_ratio_params.csv（存在すれば読み込み、なければデフォルト使用）
    mr_params_path = Path(__file__).parent / "mr_ratio_params.csv"
    if mr_params_path.exists():
        mr_params_df = pd.read_csv(mr_params_path)
        mr_ratio_params = dict(zip(mr_params_df["parameter"], mr_params_df["value"].astype(float)))
        print(f"       → MR比率パラメータ: {mr_ratio_params}")
    else:
        mr_ratio_params = None
        print("       → mr_ratio_params.csv なし → デフォルト値使用")

    # competitor_schedule.csv（競合品発売スケジュール）
    comp_path = Path(__file__).parent / "competitor_schedule.csv"
    if comp_path.exists():
        comp_df = pd.read_csv(comp_path)
        competition_schedule: Dict[str, List[Dict]] = {}
        for _, row in comp_df.iterrows():
            pid = str(row["product_id"]).strip()
            comp_ym_raw = row["launch_ym"]
            comp_launch_ym = normalize_ym(comp_ym_raw)
            competition_schedule.setdefault(pid, []).append({
                "launch_ym":    comp_launch_ym,
                "intensity":    float(row["intensity"]),
                "boost_months": int(float(row["boost_months"])),
            })
        print(f"       → 競合スケジュール: {sum(len(v) for v in competition_schedule.values())} 件読み込み")
    else:
        competition_schedule = {}
        print("       → competitor_schedule.csv なし → 競合ブーストなし")
    print(f"       → {len(product_configs)} 品目をロード")
    for cfg in product_configs:
        print(f"          {cfg.product_id}: {cfg.area}, LOE={cfg.loe_months}ヶ月 ({cfg.launch_ym}+{cfg.loe_months}m)")

    # product_info を CSV由来のデータで上書き（loe_monthsをCSVの値に統一）
    data["product_info"] = pd.DataFrame([
        {
            "product_id":          cfg.product_id,
            "launch_ym":           cfg.launch_ym,
            "loe_months":          cfg.loe_months,
            "estimated_patients":  cfg.estimated_patients,
            "num_indications":     cfg.num_indications,
            "post_loe_factor":     cfg.post_loe_factor,
        }
        for cfg in product_configs
    ])

    # ---- 3. モジュール初期化 ----
    print("[3/6] モジュール初期化...")

    target_doctor_calc = TargetDoctorCalculator(
        activity_data=data["activity_data"],
        doctor_attr=data["doctor_attr"],
        product_info=data["product_info"],
    )

    # target_doctors.csv からターゲット医師数キャッシュを設定
    # 新発売品（OVE）はキャッシュなし → _calculate_new_product が参照品から自動計算
    td_df = pd.read_csv(Path(__file__).parent / "target_doctors.csv")
    new_product_ids_set = {cfg.product_id for cfg in product_configs if cfg.is_new}
    target_doctor_calc._base_target_cache = {
        row["product_id"]: int(row["target_doctors"])
        for _, row in td_df.iterrows()
        if row["product_id"] not in new_product_ids_set
    }

    fc_sc_allocator = FCScAllocator(
        activity_set_df=data["activity_set_df"],
        target_doctor_lists=data["target_doctor_lists"],
        fc_ratios=data["fc_ratios"],   # fc_sc_ratio.csv の値を優先使用
    )

    freq_estimator = ActivityFrequencyEstimator(
        activity_data=data["activity_data"],
        product_info=data["product_info"],
        mmm_optimal_activities=data["mmm_optimal_df"],
    )

    mr_digital_estimator = MRDigitalRatioEstimator(
        decay_params_df=data["decay_params_df"],
    )

    # ----------------------------------------------------------
    # OVEランチカーブ（発売後月次浸透率）
    # FY2026-07発売 → FY2029末で目標FTE ~40 を達成する設定
    # 参照品: TRI（同CS、類似規模）
    # ----------------------------------------------------------
    # OVEランチカーブ（発売時が最大 → 徐々に減少）
    # 新製品は発売直後が最も活動量が多い（医師教育・処方獲得に集中投資）
    # 普及が進むにつれてメンテナンスモードへ移行し頻度を低下させる
    # OVEランチカーブ（FY2026-07発売, 116ヶ月 = FY2035末まで）
    # 発売直後が最大投資、以降は段階的に逓減（立ち上げ→浸透→維持モード）
    ove_ramp_up = [
        # --- 立ち上げ期（FY2026〜FY2028: 1〜30ヶ月）---
        1.00, 0.98, 0.96, 0.94, 0.92, 0.90,   # 1〜6ヶ月  (FY2026発売)
        0.88, 0.86, 0.84, 0.83, 0.82, 0.81,   # 7〜12ヶ月
        0.80, 0.79, 0.78, 0.77, 0.76, 0.75,   # 13〜18ヶ月 (FY2027)
        0.74, 0.73, 0.72, 0.72, 0.71, 0.71,   # 19〜24ヶ月
        0.70, 0.70, 0.69, 0.69, 0.69, 0.68,   # 25〜30ヶ月 (FY2028)
        # --- 浸透期（FY2028〜FY2030: 31〜54ヶ月）---
        0.68, 0.68, 0.67, 0.67, 0.67, 0.67,   # 31〜36ヶ月
        0.66, 0.66, 0.66, 0.66, 0.65, 0.65,   # 37〜42ヶ月 (FY2029)
        0.64, 0.63, 0.63, 0.62, 0.61, 0.61,   # 43〜48ヶ月 (FY2030)
        0.60, 0.59, 0.59, 0.58, 0.57, 0.57,   # 49〜54ヶ月
        # --- 維持期（FY2031〜FY2035: 55〜116ヶ月）--- 緩やかに逓減
        0.56, 0.55, 0.55, 0.54, 0.53, 0.53,   # 55〜60ヶ月 (FY2031)
        0.52, 0.51, 0.51, 0.50, 0.49, 0.49,   # 61〜66ヶ月
        0.48, 0.47, 0.47, 0.46, 0.45, 0.45,   # 67〜72ヶ月 (FY2032)
        0.44, 0.43, 0.43, 0.42, 0.41, 0.41,   # 73〜78ヶ月
        0.40, 0.40, 0.39, 0.39, 0.38, 0.38,   # 79〜84ヶ月 (FY2033)
        0.37, 0.37, 0.36, 0.36, 0.35, 0.35,   # 85〜90ヶ月
        0.35, 0.35, 0.35, 0.35, 0.35, 0.35,   # 91〜96ヶ月 (FY2034)
        0.35, 0.35, 0.35, 0.35, 0.35, 0.35,   # 97〜102ヶ月
        0.35, 0.35, 0.35, 0.35, 0.35, 0.35,   # 103〜108ヶ月 (FY2035)
        0.35, 0.35, 0.35, 0.35, 0.35, 0.35,   # 109〜114ヶ月
        0.35, 0.35,                             # 115〜116ヶ月
    ]

    # Zasoランチカーブ（FY2027-04発売, 108ヶ月 = FY2035末まで、参照品: ENT）
    zaso_ramp_up = [
        # --- 立ち上げ期（FY2027〜FY2029: 1〜30ヶ月）---
        1.00, 0.98, 0.96, 0.94, 0.92, 0.90,   # 1〜6ヶ月  (FY2027発売)
        0.88, 0.86, 0.84, 0.83, 0.82, 0.81,   # 7〜12ヶ月
        0.80, 0.79, 0.78, 0.77, 0.76, 0.75,   # 13〜18ヶ月 (FY2028)
        0.74, 0.73, 0.72, 0.72, 0.71, 0.71,   # 19〜24ヶ月
        0.70, 0.70, 0.69, 0.69, 0.69, 0.68,   # 25〜30ヶ月 (FY2029)
        # --- 浸透期（FY2030〜FY2031: 31〜54ヶ月）---
        0.68, 0.68, 0.67, 0.67, 0.67, 0.67,   # 31〜36ヶ月
        0.66, 0.66, 0.66, 0.66, 0.65, 0.65,   # 37〜42ヶ月 (FY2030)
        0.64, 0.63, 0.62, 0.61, 0.60, 0.59,   # 43〜48ヶ月 (FY2031)
        0.58, 0.57, 0.56, 0.55, 0.54, 0.53,   # 49〜54ヶ月
        # --- 維持期（FY2032〜FY2035: 55〜108ヶ月）---
        0.52, 0.51, 0.50, 0.49, 0.48, 0.47,   # 55〜60ヶ月 (FY2032)
        0.46, 0.45, 0.44, 0.43, 0.42, 0.41,   # 61〜66ヶ月
        0.40, 0.40, 0.40, 0.40, 0.40, 0.40,   # 67〜72ヶ月 (FY2033)
        0.40, 0.40, 0.40, 0.40, 0.40, 0.40,   # 73〜78ヶ月
        0.40, 0.40, 0.40, 0.40, 0.40, 0.40,   # 79〜84ヶ月 (FY2034)
        0.40, 0.40, 0.40, 0.40, 0.40, 0.40,   # 85〜90ヶ月
        0.40, 0.40, 0.40, 0.40, 0.40, 0.40,   # 91〜96ヶ月 (FY2035)
        0.40, 0.40, 0.40, 0.40, 0.40, 0.40,   # 97〜102ヶ月
        0.40, 0.40, 0.40, 0.40, 0.40, 0.40,   # 103〜108ヶ月
    ]

    # WSAランチカーブ（FY2029-04発売, 84ヶ月 = FY2035末まで、参照品: LIV）
    wsa_ramp_up = [
        # --- 立ち上げ期（FY2029〜FY2031: 1〜30ヶ月）---
        1.00, 0.98, 0.96, 0.94, 0.92, 0.90,   # 1〜6ヶ月  (FY2029発売)
        0.88, 0.86, 0.84, 0.83, 0.82, 0.81,   # 7〜12ヶ月
        0.80, 0.79, 0.78, 0.77, 0.76, 0.75,   # 13〜18ヶ月 (FY2030)
        0.74, 0.73, 0.72, 0.72, 0.71, 0.71,   # 19〜24ヶ月
        0.70, 0.70, 0.69, 0.69, 0.69, 0.68,   # 25〜30ヶ月 (FY2031)
        # --- 浸透期（FY2032〜FY2033: 31〜54ヶ月）---
        0.68, 0.68, 0.67, 0.67, 0.67, 0.67,   # 31〜36ヶ月
        0.66, 0.66, 0.66, 0.66, 0.65, 0.65,   # 37〜42ヶ月 (FY2032)
        0.63, 0.61, 0.60, 0.58, 0.57, 0.55,   # 43〜48ヶ月 (FY2033)
        0.54, 0.52, 0.51, 0.50, 0.50, 0.50,   # 49〜54ヶ月
        # --- 維持期（FY2034〜FY2035: 55〜84ヶ月）---
        0.50, 0.50, 0.50, 0.50, 0.50, 0.50,   # 55〜60ヶ月 (FY2034)
        0.50, 0.50, 0.50, 0.50, 0.50, 0.50,   # 61〜66ヶ月
        0.50, 0.50, 0.50, 0.50, 0.50, 0.50,   # 67〜72ヶ月 (FY2035)
        0.50, 0.50, 0.50, 0.50, 0.50, 0.50,   # 73〜78ヶ月
        0.50, 0.50, 0.50, 0.50, 0.50, 0.50,   # 79〜84ヶ月
    ]

    # GLOランチカーブ（FY2026-02発売, 13ヶ月 = FY2026末まで）
    # 既知の海外製品を代替品として導入するため、立ち上げ教育は不要 → フラット
    glo_ramp_up = [1.00] * 14  # 余裕をもって14ヶ月分（2026-02〜2027-03）

    calculator = FY2029FTECalculator(
        product_configs=product_configs,
        target_doctor_calc=target_doctor_calc,
        fc_sc_allocator=fc_sc_allocator,
        freq_estimator=freq_estimator,
        mr_digital_estimator=mr_digital_estimator,
        product_info=data["product_info"],
        current_activities=data["current_activities"],
        frequency_mode="lifecycle_adjusted",
        new_product_ramp_up={
            "OVE": ove_ramp_up, "Zaso": zaso_ramp_up, "WSA": wsa_ramp_up,
            "GLO": glo_ramp_up,   # 供給制限代替品: FY2026のみ活動
        },
        reference_products=csv_reference_products,  # products.csvのreference_product列
        target_months=ALL_MONTHS,  # FY2026〜2035の120ヶ月
        mr_ratio_params=mr_ratio_params,            # mr_ratio_params.csv から読み込み
        competition_schedule=competition_schedule,  # competitor_schedule.csv から読み込み
        supply_restrictions=data["supply_restrictions"],  # 供給制限（GLI FY2026等）
    )

    # ---- 4. FTE 算出（FY2026〜FY2029）----
    print("[4/6] FY2026〜FY2035 FTE算出...")

    # 新品目: OVE（FY2026発売）、Zaso（FY2027発売）、WSA（FY2029発売）
    # FTE不足をドナー品目から補う
    new_product_fte_allocator = NewProductFTEAllocator(
        decay_params_df=data["decay_params_df"],
        current_fte=data["current_fte"],
        current_mr_activity={
            pid: data["current_activities"][pid]["MR"]
            for pid in data["current_fte"]
        },
        min_fte_ratio=0.5,
    )

    # 動的FTE移動: 新品目（OVE/Zaso/WSA）の成長（ランチカーブ）に比例してドナー品目から段階的に削減
    fte_df, allocation_df = calculator.run_with_dynamic_new_product(
        new_product_ids=["OVE", "Zaso", "WSA"],
        donor_products=["GLI", "CUV", "HYQ", "INT", "TRI", "ENT",
                        "LIV", "REV", "ALC", "VYV", "VPR"],
        new_product_fte_allocator=new_product_fte_allocator,
    )

    # 正規化前の「本来必要FTE」を保存（制約なし、活動量から積み上げた真の必要値）
    raw_fte_df = fte_df.copy()
    raw_summary_fy = calculator.total_fte_by_area_fy(raw_fte_df)

    # ⑥ 新発売品ごとの発売時点FTE配分（どの品目から何FTE取るか）
    fte_col_raw = "adjusted_fte" if "adjusted_fte" in raw_fte_df.columns else "required_fte"
    new_product_launch_months = {"OVE": "2026-07", "Zaso": "2027-04", "WSA": "2029-04"}
    all_cs_donors = ["GLI", "CUV", "HYQ", "INT", "TRI", "ENT", "LIV", "REV", "ALC", "VYV", "VPR"]
    per_launch_allocations: Dict[str, pd.DataFrame] = {}
    for new_pid, launch_month in new_product_launch_months.items():
        month_df = raw_fte_df[raw_fte_df["month"] == launch_month]
        new_row = month_df[month_df["product_id"] == new_pid]
        if new_row.empty:
            continue
        launch_fte = float(new_row[fte_col_raw].iloc[0])
        # 発売時点で FTE > 0 のドナーのみ対象
        active_donors = [
            p for p in all_cs_donors
            if not month_df[month_df["product_id"] == p].empty
            and float(month_df[month_df["product_id"] == p][fte_col_raw].sum()) > 0
        ]
        if launch_fte > 0 and active_donors:
            per_launch_allocations[new_pid] = new_product_fte_allocator.allocate(
                launch_fte, active_donors
            )

    # 年度ヘッドカウント目標に正規化（CS=380, PS=45）
    # 品目間の相対比を保ちながら毎月の領域合計を目標値にスケーリング
    fte_df = normalize_fte_to_headcount(fte_df, CURRENT_MR_COUNT)

    # 月次FTEを半期（6ヶ月）ステップ関数に変換（離散的な計画値）
    fte_df = discretize_fte_semiannually(fte_df, CURRENT_MR_COUNT)

    # ROI最大化最適FTE算出（等限界収益配分）
    print("  ROI最適FTE算出中...")
    optimal_fte_df = calculate_roi_optimal_fte(
        decay_params_df=data["decay_params_df"],
        product_configs=product_configs,
        headcount_targets=CURRENT_MR_COUNT,
    )

    summary_df    = calculator.summarize_fy(fte_df)
    total_fte_df  = calculator.total_fte_by_area(fte_df)
    total_fte_fy  = calculator.total_fte_by_area_fy(fte_df)

    # 結果サマリー表示
    print("\n  --- 品目×年度別 平均FTE ---")
    pivot = summary_df.pivot_table(
        index="product_id", columns="fiscal_year",
        values="avg_required_fte",
    ).round(1)
    print(pivot.to_string())

    print("\n  --- 新品目（OVE/Zaso/WSA）FTE配分先 ---")
    print(allocation_df.to_string(index=False))

    print("\n  --- 領域×年度別 合計FTE vs 現行MR数 ---")
    print(total_fte_fy.to_string(index=False))

    # ---- 5. 感度分析 ----
    print("\n[5/6] 感度分析...")
    analyzer = FY2029SensitivityAnalyzer(
        base_calculator=calculator,
        base_target_calc=target_doctor_calc,
        mmm_freq_estimator=None,
    )
    sensitivity_results = analyzer.run_all_scenarios(
        new_product_ids=["OVE", "Zaso", "WSA"],
        donor_products=["GLI", "CUV", "HYQ", "INT", "TRI", "ENT",
                        "LIV", "REV", "ALC", "VYV", "VPR"],
        new_product_fte_allocator=new_product_fte_allocator,
    )

    # ---- 6. HTML出力 ----
    print(f"\n[6/6] HTMLレポート出力 → {OUTPUT_DIR}/")

    reporter = FY2029HTMLReporter(output_dir=str(OUTPUT_DIR))
    reporter.generate(
        fte_df=fte_df,
        summary_df=summary_df,
        allocation_df=allocation_df,
        total_fte_df=total_fte_df,
        total_fte_fy_df=total_fte_fy,
        optimal_fte_df=optimal_fte_df,
        raw_summary_fy=raw_summary_fy,
        per_launch_allocations=per_launch_allocations,
        frequency_mode="lifecycle_adjusted",
        filename="fy2029_fte_report.html",
    )

    analyzer.export_sensitivity_html(
        results=sensitivity_results,
        output_dir=str(OUTPUT_DIR),
        filename="fy2029_sensitivity.html",
    )

    # CSV出力
    fte_df.to_csv(OUTPUT_DIR / "fy2029_fte_detail.csv", index=False, encoding="utf-8-sig")
    summary_df.to_csv(OUTPUT_DIR / "fy2029_summary.csv", index=False, encoding="utf-8-sig")
    allocation_df.to_csv(OUTPUT_DIR / "fy2029_allocation.csv", index=False, encoding="utf-8-sig")
    total_fte_fy.to_csv(OUTPUT_DIR / "fy2029_area_fy.csv", index=False, encoding="utf-8-sig")
    optimal_fte_df.to_csv(OUTPUT_DIR / "fy2029_optimal_fte.csv", index=False, encoding="utf-8-sig")
    print(f"[OK] CSV出力完了")

    print("\n" + "=" * 65)
    print("FY2029 FTE算出 完了")
    print(f"レポート: {OUTPUT_DIR.resolve()}")
    print("=" * 65)

    return fte_df, summary_df, allocation_df, total_fte_df, total_fte_fy, sensitivity_results


if __name__ == "__main__":
    main()

# %%
